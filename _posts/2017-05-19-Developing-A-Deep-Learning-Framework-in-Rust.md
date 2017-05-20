---
layout: post
title: "Developing a Deep Learning Framework in Rust"
date: 2017-05-19
---

There are numerous deep learning frameworks available. Judging by the number of stars on github 
and the amount of attention that Google has given it, the most well known is probably Google's 
TensorFlow. For most applications, which framework is 'best' ultimately boils down to taste and
implementation details for the specific model one is developing. Of the handful that I've tried, the first framework that I've actually *enjoyed* using is PyTorch.

The simplest example that actually does something interesting is examples/mnist from pytorch/examples - classifying handwritten digits using a convolutional neural network (CNN):

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)
            
If *data* is a 2d tensor representing a grey scale image and *target* is the correct value and optimizer is one of the available optimizers - SGD, ADAM, RMSprop, etc. - instantiating the model and doing a single training pass is
    
        model = Net()
        ...
(eliding a number of critical details for the sake of clarity)

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
Calling model calls forward(data) followed by any registered forward and backward hooks. Inheriting from nn.Module let Net automatically construct a graph used for bookkeeping purposes: setting training state, tensor primitive type, location of tensor (CPU/GPU), serialization/deserialization, and pretty printing.

Although Python is a great language for exploration of ideas there are environments for which it is not a good fit. The ones that come first to mind are embedded and linked against native applications in a server environment. Facebook, the primary sponsor of PyTorch, uses Caffe2 (a single rather large C++ binary) for production within its Applied Machine Learning group. Although this is certainly a pragmatic choice, I would like to have something lighter weight with the flexibility of PyTorch. To this end I've started developing Rust bindings for the Torch C libraries with an aim to providing an API as close to the one provided by PyTorch as Rust will permit. I've chosen the logical and yet whimsical name torch.rs (torturous) for this DL framework. As a systems language that aims to have performance equivalent to C Rust is a much less compliant language than python so the framework does face the very real risk of being a dancing bear.

Based on the syntactic issues I've resolved far, an implementation of the Net class in the MNIST example shown above in torch.rs should look something like:

    use torchrs::functional::{max_pool2d, relu, conv2d, dropout(), dropout2d, linear, log_softmax}
    #[derive(Serialize, Deserialize, Debug, ModuleParse)]
    struct Net<'a> {
        delegate: Module<'a>,
        #[ignore]
        foo: i32, // for example purposes only
        conv1: Conv2d,
        conv2: Conv2d,
        fc1: Linear,
        fc2: Linear,
    }
    impl Net<'a> {
        pub fn new() -> Net<'a> {
            let t = Net {
            delegate: Module::new(),
            foo: -5,
            conv1: Conv2d::build(1, 10).kernel_size(5).done(),
            conv2: Conv2d::build(10, 20).kernel_size(5).done(),
            fc1: Linear::build(320, 50).done(),
            fc2: Linear::build(50, 10).done(),
            };
            t.init_module();
            t 
        }
    }
    impl <'a>ModIntf<'a> for Net<'a> {
        // forward could be implemented in one of two ways:
        // a) as a near verbatim implementation of the python version 
        fn forward(&mut self, args: &[&mut Tensor]) -> [&mut Tensor] {
            let training = self.delegate.training;
            let x = relu(max_pool2d(self.conv1(&args[0]), 2));
            let x = relu(max_pool2d(dropout2d(self.conv2(&x), training, 0.5), 2));
            let x = x.view(-1, 320);
            let x = relu(self.fc1(&x));
            let x = dropout(&x, training, 0.5)
            let x = self.fc2(&x);
            log_softmax(&x)
        }
        // b) having all functions be members of an implementation of the Tensor
        //    trait so that they can implicitly take a tensor and return a tensor
        //    in a chained method invocation
        fn forward(&mut self, args: &[&mut Tensor]) -> [&mut Tensor] {
            let training = self.delegate.training;
            args[0].conv2d(self.conv1).max_pool2d(2).relu(false).
            conv2d().dropout2d(training, 0.5).max_pool2d(2).relu(false).
            view(-1, 320).linear(self.fc1).relu(false).dropout(training, 0.5).
            linear(self.fc2).log_softmax()
        }
        fn delegate(&mut self) -> &mut Module<'a> { &mut self.delegate }
    }


While certainly more verbose than the PyTorch version, I think it actually does a reasonable job of capturing the spirit of the PyTorch API. In descending order of severity, the added complexity and syntax mismatch stem from the following design differences in Rust:
- No compositional inheritance
- No introspection
- No optional named arguments (kwargs)
- Only one mutable reference allowed at a time
- Overloading indexing operator '[]' only allows one argument

### Compositional Inheritance
As of 1.19 Rust does not support anything resembling compositional inheritance. This means that the familiar pattern:

    Base {
        fieldA;
        methodA();
        virtual methodB();
    }
    Subtype extends Base {
        fieldB;
        methodB() {...}
        methodC() {...}
    }

where *Subtype* will automatically embed *Base*, one can call *.methodA()* on an instance of *Subtype*, and all classes subtyping Base will support *.methodB()* - will not work in Rust (without abusing the Deref trait which would give the odd behavior that *\*Subtype* would only return the *Base* part of *Subtype*). Nonetheless, in Rust one can achieve a similar effect with the following idiom:

    struct Base {
        fieldA : atype,
    }
    impl Base {
        fn methodA(&self) -> atype { ... }
    }
    trait BaseIntf {
        fn delegate(&mut self) -> &mut Base;
        fn methodA(&mut self) -> atype { self.delegate().methodA() }
        fn methodB(&self) -> atype; 
    }
    struct Subtype {
        delegate: Base,
        fieldB: atype,
    }
    impl Subtype {
        fn methodC(&self) -> atype { ... }
    }
    impl BaseIntf for Subtype {
        fn delegate(&mut self) -> &mut Base { &mut self.delegate }
        fn methodB(&self) -> atype { ... }
    }

One can explicitly create the *Subtype* "is a" *Base* relationship while separately defining the common set of interfaces that classes inheriting from Base should support and thus support the semantics that OO languages support more automatically. This is why the torch.rs version of MNIST embeds a module and implements a *delegate()* method.


### Introspection
When I refer to introspection in Python I'm referring to two features: the abilitity to iterate over fields in an objects and match on type. PyTorch uses these features to populate a dict of parameters and a dict of modules to support apply() methods to allow one to pass a closure that will modify every network layer in the graph. PyTorch uses this for serialization/deserialization, pretty printing, changing training state, changing the primitive type of the tensors in layer parameters, and moving tensors to and from the GPU. In Rust we don't actually need to explicitly implement the first two in Module - we just need to implement the Serialize, Deserialize, and Debug for the structs that can't automatically derive them - namely leaf Parameter Tensors that wrap the native types, and then indicate that new layers should automatically derive those traits. However, we still need to be able to explicitly traverse the graph for the other functions. One can, with a bit of effort, support this functionality using procedural macros and attribute annotations. The "#[ignore]" annotation is an attribute that tells the ModParse derive_macro that that field does not implement the ModIntf trait so that it won't try to store it in a HashMap of Modules.

    pub struct Module<'a> {
        pub _name: &'a str,
        <...>
        _modules: HashMap<&'a str, *mut Module<'a>>,
        <...>
    }

    impl Module {
        <...>
        pub fn add_module(&mut self, module: &mut ModIntf<'a>) {
            let m = module.delegate();
            self._modules.insert(m._name, m.as_mut_ptr());
        }
        <...>
    }
    
    impl ModuleStruct for Net {
        fn init_module(&mut self) {
            self.delegate._name = "Net";
            self.delegate.add_module(&mut self.conv1);
            self.delegate.add_module(&mut self.conv2);
            self.delegate.add_module(&mut self.conv_drop);
            self.delegate.add_module(&mut self.fc1);
            self.delegate.add_module(&mut self.fc2);
        }
    }

We can now recursively iterate over the children of each layer just as PyTorch does. This requirement is why we have the #[derive(<...>, ModuleParse)] and the #[ignore] before each field that is not a Module, Parameter, or implementer of ModIntf.



### Optional Named Arguments - aka kwargs
Rust has no named optional arguments the way that Python, Ocaml, and numerous other languages do. Some on #rust say
that it would be desirable to have this but there is no consensus on the "right" way to support this. However, there
is the [builder pattern](https://aturon.github.io/ownership/builders.html) that we see used for the convolutional and 
fully connected layers.


### Rust insists that mutable references be unique
By design Rust limits the user to one mutable reference at a time if one is not using additional protections. This meant that the naive initial implementation of directly storing references to the individual fields would not actually compile. We need to either resort to unsafe code to achieve this or implement the iterator by storing just the names of fields and generating a separate function with a match statement to return a field given a string. This is why we coerce the module to a *\*mut* with its *as_mut_ptr* method. In order to hide this from client code we implement a custom iterator that functions as a wrapper to map the *\*mut* back to *&mut*.

    pub struct PtrIterMut<'a, T: 'a> {
        mod_iter: hash_map::IterMut<'a, &'a str, *mut T>,
    }
    impl<'a, T> Iterator for PtrIterMut<'a, T> {
        type Item = (&'a str, &'a mut T);
        fn next(&mut self) -> Option<Self::Item> {
            if let Some((name, t)) = self.mod_iter.next() {
                Some((name, unsafe { &mut **t as &mut T }))
            } else {
                None
            }
        }
    }
    impl<'a> Module<'a> {
        <...>
        pub fn modules_iter_mut(&mut self) -> PtrIterMut<Module<'a>> {
            PtrIterMut { mod_iter: self._modules.iter_mut() }
        }
        <...>
    }

### Overloading the index operator
The MNIST example doesn't show any direct use of Tensors, but in PyTorch one can define a 5d tensor and index in to it as you would a 5d array in C. Python can look at an arbitrary number of '[]' and pass those in as arguments to the indexing function in native code. In Rust, the Index and IndexMut traits are limited to a single '[]'. This means that indexing in to a 5d tensor can only be done with a single 5 element array: *data[[1, 2, 3, 4, 5]].

### Conclusion
This is just a quick overview of the issues that I have encountered so far. The next step is getting autograd (backpropagation support) working in one form or another. Although complex, the actual amount of code in the autograd implementation is quite small. The bulk of the sources are dedicated to interfacing with Python. This leads me to believe that rather than writing a rust wrapper for the C++ autograd code it would make more sense to simply port it to Rust. What that entails is a matter for a later post.
