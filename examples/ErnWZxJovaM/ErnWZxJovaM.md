# Introduction to the Course

00:00:00 - [Music] Good afternoon, everyone, and welcome to MIT's 6.S191. My name is Alexander Amini, and I'll be one of your instructors for the course this year, along with Ava. Together, we're really excited to welcome you to this incredible course.

00:00:27 - This is a very fast-paced and intense one-week experience that we're about to go through together. We will cover the foundations of a rapidly evolving field that has been changing significantly over the past eight years that we have taught this course at MIT. In fact, over the past decade, even before we started teaching this course, AI and deep learning have been revolutionizing many different areas of science, mathematics, physics, and more.

00:01:05 - Not long ago, we faced challenges and problems that we did not think were necessarily solvable in our lifetimes. Yet, AI is now solving these problems, often surpassing human performance. Each year that we teach this course, this particular lecture is getting harder and harder to deliver. For an introductory level course, this first lecture is supposed to cover the foundations. If you think about any other introductory course, like a 101 course in mathematics or biology, those first lectures don't change much over time. However, we are in a rapidly changing field of AI and deep learning, where even these foundational lectures are evolving quickly.

00:01:53 - Let me give you an example of how we introduced this course only a few years ago. We welcomed everyone to MIT's 6.S191, the official introductory course on deep learning taught here at MIT. Deep learning is revolutionizing many fields, from robotics to medicine and everything in between. In this course, you'll learn the fundamentals of this field and how to build incredible algorithms. In fact, the entire speech and video you just watched were not real; they were created using deep learning and artificial intelligence. In this class, you'll learn how this technology works.

00:03:05 - The surprising thing about that video, when we first created it, was how viral it went a few years ago. Within just a couple of months of teaching this course, that video garnered over a million views. People were shocked by a few things, but the main one was the realism of AI in generating content that looks and sounds hyper-realistic. When we created that video, it took us about $10,000 in compute to generate just a minute-long video. If you think about it, that's extremely expensive for computing something like that. 

00:03:47 - Today, many of you might not be as impressed by the technology because you see all the amazing things that AI and deep learning are producing now. Fast forward to today, the progress in deep learning is astonishing. People were making all kinds of exciting remarks about it when it first came out, but now this technology is common. AI is doing much more powerful things than that fun little introductory video.

00:04:19 - So, where are we now? AI is now generating content with deep learning being commoditized. Deep learning is at our fingertips, available online and on our smartphones. In fact, we can use deep learning to generate hyper-realistic pieces of media and content entirely from English language prompts, without even needing to code anymore. 

00:04:46 - Before, we had to train these models and code them to create that one-minute-long video. Today, we have models that can do that for us end-to-end, directly from English language instructions. We can ask these models to create something that the world has never seen before, like a photo of an astronaut riding a horse. These models can imagine and generate those pieces of content entirely from scratch.

00:05:09 - My personal favorite is how we can now ask these deep learning models to create new types of software. For example, we can ask them to write TensorFlow code to train a neural network. We're asking a neural network to write code to train another neural network, and our model can produce examples of functional and usable pieces of code that satisfy this English prompt, while also educating the user on what each part of the code does.

00:05:54 - You can see examples of this, and what I'm trying to show you is how far deep learning has come, even in just a couple of years since we started teaching this course. 

# Foundations of Deep Learning

00:06:05 - Going back even further, to eight years ago, the most amazing thing that you'll see in this course, in my opinion, is that what we try to do here is to teach you the foundations of all of this: how all of these different types of models are created from the ground up and how we can make all of these amazing advances possible. This way, you can also do it on your own. 

00:06:29 - As I mentioned in the beginning, this introductory course is getting harder and harder to teach every year. I don't know where the field is going to be next year, and that's my honest truth. Honestly, I can't predict where it will be even in one or two months from now, just because it's moving so incredibly fast. 

00:06:50 - What I do know is that what we will share with you in this course, as part of this one week, is going to be the foundations of all the technologies that we have seen up until this point. This knowledge will allow you to create that future for yourselves and to design brand new types of deep learning models using those fundamentals and foundations. 

00:07:13 - So, let's get started with all of that and begin to figure out how we can actually achieve all of these different pieces and learn all of these different components. We should start by really tackling the foundations from the very beginning and asking ourselves: we've heard this term. I think all of you, obviously, before you've come to this class today, have heard the term deep learning. 

00:07:34 - It's important for you to really understand how this concept of deep learning relates to all of the other pieces of science that you've learned about so far. To do that, we have to start from the very beginning and think about what intelligence is at its core—not even artificial intelligence, but just intelligence. 

00:07:54 - The way I like to think about this is that intelligence is the ability to process information, which will inform your future decision-making abilities. That's something that we as humans do every single day. Now, artificial intelligence is simply the ability for us to give computers that same ability to process information and inform future decisions. 

00:08:23 - Machine learning is simply a subset of artificial intelligence. The way you should think of machine learning is as the programming ability—or, let's say, even simpler than that: machine learning is the science of trying to teach computers how to do that processing of information and decision-making from data. 

00:08:45 - Instead of hardcoding some of these rules into machines and programming them like we used to do in software engineering classes, we are now going to try to do that processing of information and inform future decision-making abilities directly from data. 

00:09:00 - Going one step deeper, deep learning is simply the subset of machine learning that uses neural networks to do that. It uses neural networks to process raw pieces of unprocessed data and allows them to ingest all of those very large data sets to inform future decisions. 

00:09:21 - Now, that's exactly what this class is really all about. If I had to summarize this class in just one line, it's about teaching machines how to process data, process information, and inform decision-making abilities from that data, learning it from that data. 

# Course Structure and Labs

00:09:37 - This program is split between two different parts. You should think of this class as being captured with both technical lectures, which for example this is one part of, as well as software labs. We'll have several new updates this year, as I mentioned earlier, just covering the rapid changes and advances in AI. 

00:10:00 - Especially in some of the later lectures, you're going to see those updates. The first lecture today is going to cover the foundations of neural networks themselves, starting with the building blocks of every single neural network, which is called the perceptron. 

00:10:15 - Finally, we'll go through the week and conclude with a series of exciting guest lectures from industry-leading sponsors of the course. On the software side, after every lecture, you'll also get software experience and project-building experience to be able to take what we teach in lectures and actually deploy them in real code, producing based on the learnings that you find in this lecture. 

00:10:42 - At the very end of the class, from the software side, you'll have the ability to participate in a really fun day, which is the project pitch competition. It's kind of like a Shark Tank-style competition for all of the different projects from all of you, where you can win some really awesome prizes. 

00:10:57 - Let's step through that a little bit briefly. This is the syllabus part of the lecture. Each day, we'll have dedicated software labs that will mirror all of the technical lectures we go through, helping you reinforce your learnings. These are coupled with prizes for the top-performing software solutions that emerge in the course.

00:11:24 - Class will start today with Lab One, which will focus on music generation. You will learn how to build a neural network that can learn from a variety of musical songs, listen to them, and then compose brand new songs in that same genre. Tomorrow, Lab Two will cover computer vision, where you will learn about facial detection systems. You will build a facial detection system from scratch using convolutional neural networks. You'll learn what that means tomorrow, and you'll also learn how to debias and remove the biases that exist in some of these facial detection systems, which is a significant problem for the state-of-the-art solutions that exist today.

# Course Overview and Resources

00:12:07 - Finally, a brand new lab at the end of the course will focus on large language models. You will take a multi-billion parameter large language model and fine-tune it to build an assistive chatbot, evaluating a set of cognitive abilities ranging from mathematics to scientific reasoning and logical abilities. At the very end, there will be a final project pitch competition for up to five minutes per team, and all of these are accompanied by great prizes. There will definitely be a lot of fun to be had throughout the week.

00:12:42 - There are many resources to help with this class, which you will see posted here. You don't need to write them down because all of the slides are already posted online. Please post to Piazza if you have any questions. We have an amazing team that is helping teach this course this year, and you can reach out to any of us if you have any questions. Piazza is a great place to start. I, along with AA, will be the two main lecturers for this course, especially from Monday through Wednesday. We will also be hearing some amazing guest lectures in the second half of the course, which you will definitely want to attend, as they cover the state-of-the-art aspects of deep learning happening in industry outside of academia.

00:13:27 - I want to give a huge thanks to all of our sponsors, without whose support this course, like every year, would not be possible. 

# Introduction to Deep Learning

00:13:36 - Now, let's start with the fun stuff, which is my favorite part of the course: the technical parts. Let's begin by asking ourselves a question: why do we care about all of this? Why do we care about deep learning? Why did you all come here today to learn and listen to this course?

00:13:56 - To understand this, we need to go back a bit to see how machine learning used to be performed. Traditionally, machine learning would define a set of features, which you can think of as a set of things to look for in an image or in a piece of data. Usually, these features are hand-engineered, meaning humans would have to define them themselves. The problem with this approach is that it tends to be very brittle in practice, simply due to the nature of human-defined features.

00:14:25 - The key idea of deep learning, and what you will learn throughout this entire week, is this paradigm shift: moving away from hand-engineering features and rules that computers should look for, and instead trying to learn them directly from raw pieces of data. What are the patterns we need to identify in datasets so that we can make interesting decisions and actions based on them?

00:14:55 - For example, if we wanted to learn how to detect faces, we might consider how we would detect faces ourselves. When looking at a picture, we look for particular patterns: eyes, noses, and ears. When these elements are composed in a certain way, we deduce that it is a face. Computers do something very similar; they must understand what patterns to look for—what the eyes, noses, and ears of those pieces of data are—and then detect and predict from them.

00:15:34 - The interesting thing about deep learning is that the foundations for doing exactly what I just mentioned—picking out the building blocks and features from raw pieces of data—have existed for many decades. The question at this point is: why are we studying this now? Why is all of this exploding with so many great advances?

00:16:02 - There are three main reasons. First, the data available to us today is significantly more pervasive. These models are hungry for data, and you will learn more about this in detail. We are living in a world where data is more abundant than it has ever been in our history.

00:16:24 - Secondly, these algorithms are massively compute-hungry and highly parallelizable. This means that they have greatly benefited from compute hardware that is also capable of being parallelized. The particular name of that hardware is called a GPU. 

00:16:39 - GPUs can run parallel processing streams of information and are particularly amenable to deep learning algorithms. The abundance of GPUs and that compute hardware has also pushed forward what we can do in deep learning. 

00:16:54 - Finally, the last piece is the software. It is the open-source tools that are really used as the foundational building blocks for deploying and building all of these underlying models that you're going to learn about in this course. Those open-source tools have become extremely streamlined, making it very easy for all of us to learn about these technologies within an amazing course like this. 

# Understanding Perceptrons

00:17:24 - So, let's start now with understanding the fundamental building block of a neural network. That building block is called a perceptron. Every single perceptron, and every single neural network, is built up of multiple perceptrons. You're going to learn how those perceptrons compute information themselves and how they connect to these much larger billion-parameter neural networks. 

00:17:52 - The key idea of a perceptron, or even simpler, think of it as a single neuron. A neural network is composed of many neurons, and a perceptron is just one neuron. The idea of a perceptron is actually extremely simple, and I hope that by the end of today, this idea and the processing of a perceptron becomes very clear to you. 

00:18:13 - Let's start by talking about the forward propagation of information through a single neuron. Single neurons ingest information; they can actually ingest multiple pieces of information. Here, you can see this neuron taking input from three pieces of information: X1, X2, and XM. We define the set of inputs as X1 through XM, and each of these inputs is going to be element-wise multiplied by a particular weight, denoted by W1 through WM. 

00:18:47 - This means that there is a corresponding weight for every single input, and you should think of each weight as being assigned to that input. The weights are part of the neuron itself. Now, you multiply all of these inputs with their weights together and then add them up. We take this single number after that addition and pass it through what's called a nonlinear activation function to produce your final output, which we are calling Y. 

00:19:25 - Now, what I just said is not entirely correct. I missed one critical piece of information: the bias term. The bias term allows your neuron to shift its activation function horizontally on the x-axis. On the right side, you can see a diagram illustrating mathematically the single equation that I talked through conceptually. 

00:19:48 - Now, you can see it mathematically written down as one single equation. We can rewrite this using linear algebra, using vectors and dot products. Our inputs are described by a capital X, which is simply a vector of all of our inputs X1 through XM. Our weights are described by a capital W, which is W1 through WM. 

00:20:13 - The input is obtained by taking the dot product of X and W. That dot product performs the element-wise multiplication and then sums all of the element-wise multiplications. The missing piece is that we now add the bias term, which we are calling W0. Then, we apply the nonlinearity, denoted as Z or G. 

# Activation Functions in Neural Networks

00:20:40 - I've mentioned this nonlinearity a few times. Let's dig into it a little bit more to understand what this activation function is doing. I said a couple of things about it: it is a nonlinear function. One commonly used activation function is called the sigmoid function, which you can see here on the bottom right-hand side of the screen. 

00:21:09 - The sigmoid function is very commonly used because it takes any real number as input. The x-axis is infinite, both positive and negative, but on the y-axis, it squashes every input X into a number between 0 and 1. It is a very common choice for things like probability distributions, especially if you want to convert your answers into probabilities or teach a neuron to learn a probability distribution. 

00:21:36 - In fact, there are many different types of nonlinear activation functions that are used in neural networks. Here are some common ones. Throughout this presentation, you'll see these little TensorFlow icons. In fact, throughout the entire course, you'll see these TensorFlow icons at the bottom, which basically allow you to relate some of the foundational knowledge that we're teaching in the lectures to some of the software labs. This might provide a good starting point for a lot of the pieces that you have to do later on in the software parts of the class.

00:22:07 - The sigmoid activation function, which we talked about in the last slide, is shown on the left-hand side. This function is very popular because of its probability distributions; it squashes everything between zero and one. However, you will also see two other very common types of activation functions in the middle and on the right-hand side. The other very common activation function, which is now the most popular, is on the far right-hand side. It is called the ReLU activation function, or the Rectified Linear Unit. Essentially, it is linear everywhere except at x equals zero, where there is a nonlinearity—a kind of step or break discontinuity. 

00:22:44 - The benefit of this function is that it is very easy to compute. It still has the nonlinearity that we need, and we will talk about why we need it in a moment. It is very fast, as it consists of just two linear functions combined piecewise.

# Importance of Nonlinearities

00:22:59 - Now, let's discuss why we need a nonlinearity in the first place. Why not just deal with a linear function that we pass all of these inputs through? The point of the activation function is to introduce nonlinearities. We want our neural network to deal with nonlinear data because the world is extremely nonlinear. This is important because, if you think of real-world data sets, this is just the way they are. 

00:23:37 - For instance, if you look at data sets with green and red points, and I ask you to build a neural network that can separate the green and red points, we actually need a nonlinear function to do that. We cannot solve this problem with a single line. In fact, if we used linear functions as our activation function, no matter how big our neural network is, it would still be a linear function. Linear functions combined with linear functions remain linear. Therefore, no matter how deep or how many parameters your neural network has, the best it could do to separate these green and red points would look like a straight line.

00:24:14 - Adding nonlinearities allows our neural networks to be smaller by making them more expressive and capable of capturing more complexities in the data sets. This ultimately makes them much more powerful.

# Understanding Neural Networks: Basics and Components

00:24:30 - Let's understand this with a simple example. Imagine I give you a trained neural network. What does it mean to have a trained neural network? It means I am giving you the weights. Here, let's say the bias term \( w_0 \) is going to be one, and our weight vector \( W \) is going to be \( [3, 2] \). These are just the weights of your trained neural network. 

00:24:56 - This network has two inputs, \( X_1 \) and \( X_2 \). If we want to get the output of this neural network, all we have to do is perform the same steps we discussed before: take the dot product of the inputs with the weights, add the bias, and apply the nonlinearity. These are the three components that you really have to remember as part of this class: dot product, add the bias, and apply a nonlinearity. This process will keep repeating for every single neuron.

00:25:30 - After that happens, that neuron will output a single number. Now, let's take a look at what's inside that nonlinearity. It is simply a weighted combination of those inputs with the weights. If we look at what's inside \( G \), it is a weighted combination of \( X \) and \( W \), added with a bias, which will produce a single number.

00:25:59 - However, for any input that this model could see, what this really represents is a two-dimensional line because we have two parameters in this model. We can actually plot that line and see exactly how this neuron separates points on these axes between \( X_1 \) and \( X_2 \). We can visualize its entire space by plotting the line that defines this neuron.

00:26:33 - Here, we are plotting when that line equals zero. If I give you a new data point, say \( X_1 = -1 \) and \( X_2 = 2 \), which is just an arbitrary point in this two-dimensional space, we can plot that point. Depending on which side of the line it falls on, it tells us the answer we are looking for. 

00:26:56 - The result will indicate what the sign of the answer is going to be and also what the answer itself is. If we follow the equation written at the top here and plug in -1 and 2, we're going to get \( 1 - 3 - 4 \), which equals -6. When I put that into my nonlinearity \( G \), I'm going to get a final output of 0.2. Don't worry about the final output; that's just going to be the output for that signal function. 

00:27:27 - The important point to remember here is that the sigmoid function actually divides the space into these two parts. It squashes everything between 0 and 1, but it divides it implicitly by everything less than 0.5 and greater than 0.5, depending on whether \( x \) is less than zero or greater than zero. So, depending on which side of the line you fall on—remember, the line is when \( x \) equals \( z \)—the input to the sigmoid is zero. If you fall on the left side of the line, your output will be less than 0.5 because you're on the negative side of the line. If your input is on the right side of the line, your output will be greater than 0.5. 

00:28:12 - Here, we can actually visualize this space. This is called the feature space of a neural network. We can visualize it in its entirety; we can totally visualize and interpret this neural network. We can understand exactly what it's going to do for any input that it sees. However, this is a very simple neuron; it's not a neural network, it's just one neuron. Even more than that, it's a very simple neuron with only two inputs. 

00:28:42 - In reality, the types of neurons that you're going to be dealing with in this course are going to be neurons and neural networks with millions or even billions of parameters. Here, we only have two weights, \( W_1 \) and \( W_2 \), but today's neural networks have billions of these parameters. So, drawing these types of plots that you see here obviously becomes a lot more challenging; it's actually not possible. 

# Building Neural Networks: From Perceptrons to Layers

00:29:09 - Now that we have some of the intuition behind a perceptron, let's start building neural networks and seeing how all of this comes together. Let's revisit that previous diagram of a perceptron. If there's only one thing to take away from this lecture right now, it's to remember how a perceptron works. That equation of a perceptron is extremely important for every single class that comes after today. 

00:29:33 - There are only three steps: take the dot product with the inputs, add a bias, and apply your nonlinearity. Let's simplify the diagram a little bit. I'll remove the weight labels from this picture, and now you can assume that if I show a line, every single line has an associated weight that comes with that line. I'll also remove the bias term for simplicity; assume that every neuron has that bias term, so I don't need to show it. 

00:30:02 - Now note that the result here, now calling it \( Z \), which is just the dot product plus bias before the nonlinearity, is going to be linear. First of all, it's just a weighted sum of all those pieces; we have not applied the nonlinearity yet. But our final output is just going to be \( G(Z) \), which is the activation function or nonlinear activation function applied to \( Z \). 

00:30:28 - If we want to step this up a little bit more and say, what if we had a multi-output function? Now we don't just have one output, but let's say we want to have two outputs. We can just have two neurons in this network. Every neuron sees all of the inputs that came before it, but now you see the top neuron is going to be predicting an answer, and the bottom neuron will predict its own answer. 

00:30:56 - Importantly, one thing you should really notice here is that each neuron has its own weights. Each neuron has its own lines that are coming into just that neuron, so they're acting independently, but they can later on communicate if you have another layer. 

00:31:18 - Let's start now by initializing this process a bit further and thinking about it more programmatically. What if we wanted to program this neural network ourselves from scratch? Remember that equation I told you? It didn't sound very complex: take a dot product, add a bias (which is a single number), and apply nonlinearity. Let's see how we would actually implement something like that. 

00:31:42 - To define the layer, we're now going to call this a layer, which is a collection of neurons. We have to first define how that information propagates through the network. We can do that by creating a call function. 

00:31:56 - First, we're going to actually define the weights for that network. Remember, every neuron has weights and a bias. Let's define those first. We're going to create the call function to actually see how we can pass information through that layer. This is going to take us inputs, which is like what we previously called \( X \). It's the same story that we've been seeing this whole class. 

00:32:26 - We're going to matrix multiply or take a dot product of our inputs with our weights, add a bias, and then apply a nonlinearity. It's really that simple. We've now created a single-layer neural network. This line, in particular, is the part that allows us to be a powerful neural network, maintaining that nonlinearity. 

00:32:54 - The important thing here is to note that modern deep learning toolboxes and libraries already implement a lot of these for you. It's important for you to understand the foundations, but in practice, all of that layer architecture and all that layer logic is actually implemented in tools like TensorFlow and PyTorch through a dense layer. 

00:33:16 - Here, you can see an example of calling or creating a dense layer with two neurons, allowing it to feed in an arbitrary set of inputs. Here, we're seeing these two neurons in a layer being fed three inputs. In code, it's only reduced down to this one line of TensorFlow code, making it extremely easy and convenient for us to use these functions and call them. 

# Introduction to Neural Networks

00:33:46 - Now, let's look at our single-layer neural network. This is where we have one layer between our input and our outputs. We're slowly and progressively increasing the complexity of our neural network so that we can build up all of these building blocks. This layer in the middle is called a hidden layer, obviously because you don't directly observe it. You do observe the two input and output layers, but your hidden layer is just a neuron layer that you don't directly observe. 

00:34:18 - It gives your network more capacity and more learning complexity. Since we now have a transformation function from inputs to hidden layers and hidden layers to output, we now have a two-layered neural network. This means that we also have two weight matrices. We don't have just \( W_1 \), which we previously had to create this hidden layer, but now we also have \( W_2 \), which does the transformation from the hidden layer to the output layer. 

00:34:44 - Yes, what happens to the nonlinearity in hidden layers? You have just linear, so it's not a perceptron or not. Yes, every hidden layer also has a nonlinearity accompanied with it. That's a very important point because if you don't have that perceptron, then it's just a very large linear function followed by a final nonlinearity at the very end. You need that cascading and overlapping application of nonlinearities that occur throughout the network. 

00:35:17 - Awesome! Now, let's zoom in and look at a single unit in the hidden layer. Take this one for example; let's call it \( Z_2 \). It's the second neuron in the first layer. It's the same perception that we saw before. We compute its answer by taking a dot product of its weights with its inputs, adding a bias, and then applying a nonlinearity. 

00:35:40 - If we took a different hidden node, like \( Z_3 \), the one right below it, we would compute its answer exactly the same way that we computed \( Z_2 \), except its weights would be different from the weights of \( Z_2 \). Everything else stays exactly the same; it sees the same inputs. However, I'm not going to actually show \( Z_3 \) in this picture, and now this picture is getting a little bit messy. 

00:36:00 - So, let's clean things up a little bit more. I'm going to remove all the lines now and replace them with these boxes, these symbols that will denote what we call a fully connected layer. These layers now denote that everything in our input is connected to everything in our output, and the transformation is exactly as we saw before: dot product, bias, and nonlinearity. 

00:36:22 - Again, in code, to do this is extremely straightforward. With the foundations that we've built up from the beginning of the class, we can now just define two of these dense layers: our hidden layer on line one with \( n \) hidden units, and then our output layer with two output units. 

00:36:41 - Does that mean the nonlinearity function must be the same between layers? The nonlinearity function does not need to be the same through each layer. Oftentimes, it is because of convenience. There are some cases where you would want it to be different as well, especially in lecture two, where you're going to see nonlinearities be different even within the same layer, let alone different layers. However, unless for a particular reason, generally, the convention is that there's no need to keep them differently. 

# Building a Deep Neural Network

00:37:12 - Now, let's keep expanding our knowledge a little bit more. If we now want to make a deep neural network, not just a neural network like we saw in the previous slide, now it's deep. All that means is that we're going to stack these layers on top of each other, one by one, creating a hierarchical model. The final output is now going to be computed by going deeper and deeper into the neural network. 

00:37:39 - This, in code, again follows the exact same story as before, just cascading these TensorFlow layers on top of each other and going deeper into the network. 

00:37:50 - Okay, so now this is great because we have at least a solid foundational understanding of how to not only define a single neuron but also how to define an entire neural network. You should be able to explain at this point or understand how information goes from input through an entire neural network to compute an output. 

# Applying Neural Networks to Real Problems

00:38:11 - Now, let's look at how we can apply these neural networks to solve a very real problem that I'm sure all of you care about. Here’s a problem: how do we want to build an AI system to learn to answer the following question: will I pass this class? I'm sure all of you are really worried about this question. 

00:38:33 - To do this, let's start with a simple input feature model. The two features that we will concern ourselves with are going to be, number one, how many lectures you attend, and number two, how many hours you spend on your final project. 

00:38:48 - Let’s look at some of the past years of this class. We can actually observe how different people have lived in this space, right, between how many lectures they attended and how much time they spent on their final project. You can actually see that every point represents a person. The color of that point indicates whether they passed or failed the class. You can visualize this feature space that we talked about before. 

00:39:19 - Now, you fall right here; you're the point 45, right in between this feature space. You've attended four lectures and spent five hours on the final project. You want to build a neural network to determine, given everyone else in the class that we’ve seen from all of the previous years, what is your likelihood of passing or failing this class. 

00:39:47 - So, let’s do it. We now have all of the building blocks to solve this problem using a neural network. We have two inputs: the number of lectures you attend and the number of hours you spend on your final project, which are four and five. We can pass those two inputs to our X1 and X2 variables. 

00:40:05 - These are fed into this single-layered, single-hidden-layer neural network that has three hidden units in the middle. We can see that the final predicted output probability for you to pass this class is 0.1, or 10%. 

00:40:21 - This is a very bleak outcome; it's not a good outcome. The actual probability is one. By attending four out of the five lectures and spending five hours on your final project, you actually lived in a part of the feature space that was very positive. It looked like you were going to pass the class. 

# Training Neural Networks

00:40:37 - So, what happened here? Does anyone have any ideas? Why did the neural network get this so terribly wrong? Right, it's not trained. Exactly. This neural network is not trained; we haven't shown it any of that data, the green and red data. 

00:40:51 - You should really think of neural networks like babies. Before they see data, they haven't learned anything. There’s no expectation that we should have for them to be able to solve any of these types of problems before we teach them something about the world. 

00:41:06 - So, let’s teach this neural network something about the problem first. To train it, we first need to tell our neural network when it's making bad decisions. We need to teach it, really train it to learn, just like how we as humans learn in some ways. 

00:41:22 - We have to inform the neural network when it gets the answer incorrect so that it can learn how to get the answer correct. The closer the answer is to the ground truth—so, for example, the actual value for you passing this class was a probability of one, or 100%, but it predicted a probability of 0.1—we compute what's called a loss. 

00:41:45 - The closer these two things are together, the smaller your loss should be, and the more accurate your model should be. 

00:41:53 - Let’s assume that we have data not just from one student, but now we have data from many students. Many students have taken this class before, and we can plug all of them into the neural network and show them to this system. 

00:42:06 - Now, we care not only about how the neural network did on just this one prediction, but we care about how it predicted for all of these different people that the neural network has seen in the past during this training and learning process. 

00:42:21 - When training the neural network, we want to find a network that minimizes the empirical loss between our predictions and those ground truth outputs. We’re going to do this on average across all of the different inputs that the model has seen. 

# Introduction to Binary Classification and Loss Functions

00:42:38 - If we look at this problem of binary classification, right, between yeses and noes—will I pass the class or will I not pass the class?—it's a zero or one probability. We can use what is called the softmax function or the softmax cross-entropy function to inform us if this network is getting the answer correct or incorrect. The softmax cross-entropy function, think of this as an objective function; it's a loss function that tells our neural network how far away these two probability distributions are. 

00:43:15 - The output is a probability distribution, and we're trying to determine how bad of an answer the neural network is predicting so that we can give it feedback to get a better answer. 

00:43:25 - Now, let's suppose instead of training or predicting a binary output, we want to predict a real-valued output, like any number that can take any value, plus or minus infinity. For example, if you wanted to predict the grade that you get in a class, it doesn't necessarily need to be between 0 and 1 or 0 and 100. You could now use a different loss function in order to produce that value because our outputs are no longer a probability distribution. 

00:43:56 - For example, what you might do here is compute a mean squared error loss function between your true value or true grade of the class and the predicted grade. These are two numbers; they're not probabilities necessarily. You compute their difference, square it to look at the distance between the two—an absolute distance where the sign doesn't matter—and then you can minimize this value.

00:44:19 - Now, let's put all of this loss information together with the problem of finding our network into a unified problem and a unified solution to actually train our neural network. We know that we want to find a neural network that will solve this problem on all this data on average. This means effectively that we're trying to find what the weights for our neural network are. What is this big vector W that we talked about earlier in the lecture? We want to compute this vector W based on all of the data that we have seen.

00:45:03 - Now, the vector W is also going to determine what the loss is. Given a single vector W, we can compute how bad this neural network is performing on our data. What is the loss? What is this deviation from the ground truth of our network based on where it should be? 

00:45:27 - Remember that W is just a group of a bunch of numbers; it's a very big list of numbers, a list of weights for every single layer and every single neuron in our neural network. So, it's just a very big list or a vector of weights. We want to find that vector based on a lot of data. That's the problem of training a neural network. 

00:45:49 - Our loss function is just a simple function of our weights. If we have only two weights in our neural network, like we saw earlier in the slide, then we can plot the loss landscape over this two-dimensional space. We have two weights, W1 and W2, and for every single configuration or setting of those two weights, our loss will have a particular value, which here we're showing as the height of this graph. 

00:46:16 - For any W1 and W2, what is the loss? What we want to do is find the lowest point—what is the best loss? Where are the weights such that our loss will be as good as possible? The smaller the loss, the better. So, we want to find the lowest point in this graph.

00:46:35 - Now, how do we do that? The way this works is we start somewhere in this space. We don't know where to start, so let's pick a random place to start. From that place, let's compute what's called the gradient of the landscape at that particular point. This is a very local estimate of where the slope is increasing at my current location. 

00:47:03 - That informs us not only where the slope is increasing but, more importantly, where the slope is decreasing. If I negate the direction and go in the opposite direction, I can actually step down into the landscape and change my weights such that I lower my loss. 

00:47:21 - So, let's take a small step, just a small step in the opposite direction of the part that's going up. We'll keep repeating this process: we'll compute a new gradient at that new point, take another small step, and keep doing this over and over again until we converge at what's called a local minimum. 

00:47:39 - Based on where we started, it may not be a global minimum of everywhere in this loss landscape, but let's find ourselves now in a local minimum. We're guaranteed to converge by following this very simple algorithm at a local minimum. 

00:47:52 - Now, let's summarize this algorithm. This algorithm is called gradient descent. Let's summarize it first in pseudo code, and then we'll look at it in actual code in a second. 

00:48:02 - There are a few steps. The first step is to initialize our location somewhere randomly in this weight space. We compute the gradient of our loss with respect to our weights. Then, we take a small step in the opposite direction, and we keep repeating this in a loop over and over again. We say we keep doing this until convergence, until we stop moving basically, and our network finds where it's supposed to end up. 

00:48:38 - We'll talk about this small step, which we keep calling a small step. We'll discuss that a bit more in a later part of this lecture. For now, let's also very quickly show the analogous part in code as well, and it mirrors very nicely. We randomly initialize our weights; this happens every time you train a neural network. Then, we have a loop where we compute the loss at that location, compute the gradient (which tells us which way is up), negate that gradient, multiply it by what's called the learning rate (LR), and then we take a direction in that small step. 

# Gradient Descent and Backpropagation

00:49:23 - Let's take a deeper look at this term here; this is called the gradient. It tells us which way is up in that landscape, and it also tells us how our loss is changing as a function of all of our weights. However, I have not yet told you how to compute this, so let's talk about that process. This process is called backpropagation. We'll go through this very briefly, starting with the simplest neural network possible. 

00:49:55 - We already saw the simplest building block, which is a single neuron. Now, let's build the simplest neural network, which is just a one-neuron neural network. It has one hidden neuron that goes from input to hidden neuron to output, and we want to compute the gradient of our loss with respect to this weight W2. 

00:50:13 - I'm highlighting it here; we have two weights. Let's compute the gradient first with respect to W2. This tells us how much a small change in W2 affects our loss. Does our loss go up or down if we move W2 a little bit in one direction or another? Let's write out this derivative. We can start by applying the chain rule backwards from the loss through the output. 

00:50:37 - Specifically, we can decompose this derivative, this gradient, into two parts. The first part decomposes DJ/dW2 into DJ/dY, which is our output, multiplied by dY/dW2. This is all possible because Y is only dependent on the previous layer. 

00:51:12 - Now, let's suppose we don't want to do this for W2 but for W1. We can use the exact same process, but now it's one step further. We'll replace W2 with W1 and need to apply the chain rule yet again to decompose the problem further. We propagate our old gradient that we computed for W2 all the way back one more step to the weight that we're interested in, which in this case is W1. 

00:51:39 - We keep repeating this process, propagating these gradients backwards from output to input to compute ultimately what we want in the end: the derivative of every weight. This tells us how much a small change in every single weight in our network affects the loss. Does our loss go up or down if we change this weight a little bit in one direction or a little bit in the other direction? 

00:52:05 - Yes, I think you used the term "neuron" and "perceptron." Is there a functional difference? Neuron and perceptron are the same. Typically, people say "neural network," which is why a single neuron has also gained popularity. However, originally, a perceptron is the formal term; the two terms are identical. 

00:52:26 - Now that we've covered a lot, we've discussed the forward propagation of information through a neuron and through a neural network, and we've covered the backpropagation of information to understand how we should change every single one of those weights in our neural network to improve our loss. 

00:52:44 - So, that was the backpropagation algorithm. In theory, it's actually pretty simple; it's just a chain rule. There's nothing more than just the chain rule. The nice part is that deep learning libraries actually do this for you, computing backpropagation for you, so you don't have to implement it yourself, which is very convenient. 

00:53:02 - However, it's important to touch on the practical aspects. Even though the theory of backpropagation is not that complicated, let's consider some insights for your own implementations when you want to implement these neural networks. 

# Understanding Neural Network Loss Landscapes

00:53:14 - What are some insights? The implementation of neural networks in practice is a completely different story; it's not straightforward at all. In practice, it is very difficult and usually very computationally intensive to execute this backpropagation algorithm. 

00:53:30 - Here's an illustration from a paper that came out a few years ago, which attempted to visualize a very deep neural network's loss landscape. Previously, we had another depiction of how a neural network would look in a two-dimensional landscape. However, real neural networks are not two-dimensional; they exist in hundreds, millions, or even billions of dimensions. 

00:53:55 - What would those loss landscapes look like? You can actually try some clever techniques to visualize them. This particular paper attempted to do that, and it turns out that they look extremely messy. The important thing is that if you execute this algorithm and start in a bad place, depending on your neural network, you may not actually end up at the global solution. Therefore, your initialization matters a lot. You need to traverse these local minima and try to help find the global minima. 

00:54:24 - Moreover, you need to construct neural networks that have loss landscapes that are much more amenable to optimization than the one depicted in the paper. This is a very bad loss landscape. There are some techniques that we can apply to our neural networks to smooth out their loss landscape and make them easier to optimize. 

00:54:41 - Recall that update equation we talked about earlier with gradient descent. There is a parameter here that we didn't discuss; we described this as the small step that you could take. It's a small number that multiplies with the direction, which is your gradient. It tells you that you are not going to go all the way in this direction but will take a small step instead. 

00:55:07 - In practice, even setting this value can be rather difficult. If we set the learning rate too small, the model can get stuck in these local minima. For instance, it might start and then get stuck in a local minimum, converging very slowly. Even if it doesn't get stuck, if the learning rate is too large, it can overshoot, and in practice, it may even diverge and explode, meaning you never actually find any minima. 

00:55:33 - Ideally, we want to use learning rates that are not too small and not too large. They should be large enough to avoid those local minima but small enough so that they won't diverge and will still find their way into the global minima. 

00:55:52 - Something like this is what you should intuitively have in mind: a learning rate that can overshoot the local minima but eventually find itself in a better minimum and then stabilize there. 

00:56:02 - So, how do we actually set these learning rates in practice? What does that process look like? 

00:56:09 - Idea number one is very basic: try a bunch of different learning rates and see what works. This is not a bad process in practice; it's one of the methods that people use. However, let's see if we can do something smarter and design algorithms that can adapt to the landscapes. 

00:56:30 - In practice, there is no reason why the learning rate should be a single number. Can we have learning rates that adapt to the model, the data, the landscapes, and the gradients that it encounters? This means that the learning rate may actually increase or decrease as a function of the gradients in the loss function, or based on how fast we are learning, among many other options. 

00:56:55 - There are many widely used procedures or methodologies for setting the learning rate. During your labs, we encourage you to try out some of these different ideas for various types of learning rates. Experiment with what happens when you increase or decrease your learning rate; you'll see very striking differences. 

00:57:28 - Now, why not just find the absolute minimum? A few things to consider: first, it's not a closed space. Every weight can be plus or minus up to infinity. Even if it were a one-dimensional neural network with just one weight, it would not be a closed space. In practice, it's even worse than that because you have billions of dimensions. 

00:57:50 - So, not only is your support system in one dimension infinite, but you now have billions of infinite dimensions or billions of infinite support spaces. It's not practical to search every possible weight in your neural network configuration. Testing every possible weight that this neural network could take is simply not feasible, even for a very small neural network.

# Optimizers and Learning Rates in Practice

00:58:19 - In practice, in your labs, you can really try to put all of this information into practice, which defines your model. Number one, right here, defines your optimizer, which previously we denoted as this gradient descent optimizer. Here, we're calling it stochastic gradient descent or SGD. We'll talk about that more in a second. 

00:58:39 - Also, note that your optimizer, which here we're calling SGD, could be any of these adaptive optimizers. You can swap them out, and you should swap them out. You should test different things here to see the impact of these different methods on your training procedure, and you'll gain very valuable intuition for the different insights that will come with that as well.

# Batching Data for Efficient Training

00:58:59 - I want to continue very briefly, just for the end of this lecture, to talk about tips for training neural networks in practice and how we can focus on this powerful idea of what's called batching data. This means not seeing all of your data but now talking about a topic called batching. 

00:59:20 - To do this, let's very briefly revisit the gradient descent algorithm. The gradient computation, the backpropagation algorithm, is a very computationally expensive operation. It's even worse because we previously described it in a way where we would have to compute it over a summation of every single data point in our entire data set. That's how we defined it with the loss function; it's an average over all of our data points, which means that we're summing over all of our data points' gradients. 

00:59:49 - In most real-life problems, this would be completely infeasible to do because our data sets are simply too big, and the models are too large to compute those gradients on every single iteration. Remember, this isn't just a one-time thing; it's every single step that you do. You keep taking small steps, so you keep needing to repeat this process.

01:00:09 - Instead, let's define a new gradient descent algorithm called SGD, stochastic gradient descent. Instead of computing the gradient over the entire data set, now let's just pick a single training point and compute that one training point's gradient. The nice thing about that is that it's much easier to compute that gradient; it only needs one point. The downside is that it's very noisy; it's very stochastic since it was computed using just that one example. So, you have that tradeoff that exists.

01:00:43 - What's the middle ground? The middle ground is to take not one data point and not the full data set, but a batch of data. This is called a mini-batch. In practice, this could be something like 32 pieces of data, which is a common batch size. This gives us an estimate of the true gradient. You approximate the gradient by averaging the gradient of these 32 samples. 

01:01:06 - It's still fast because 32 is much smaller than the size of your entire data set, but it's pretty quick now. It's still noisy, but it's okay usually in practice because you can still iterate much faster. Since the batch size is normally not that large—again, think of something like in the tens or the hundreds of samples—it's very fast to compute this in practice compared to regular gradient descent. 

01:01:28 - It's also much more accurate compared to stochastic gradient descent. The increase in accuracy of this gradient estimation allows us to converge to our solution significantly faster as well. It's not only about the speed; it's just about the increase in accuracy of those gradients that allows us to get to our solution much faster. This ultimately means that we can train much faster as well, and we can save compute.

01:01:52 - The other really nice thing about mini-batches is that they allow for parallelizing our computation. This was a concept that we had talked about earlier in the class, and here's where it's coming in. We can split up those batches. For instance, if our batch size is 32, we can split them up onto different workers. Different parts of the GPU can tackle those different parts of our data points. This can allow us to achieve even more significant speed-ups using GPU architectures and GPU hardware.

# Understanding Overfitting and Underfitting

01:02:28 - Finally, the last topic I want to talk about before we end this lecture and move on to lecture number two is overfitting. Overfitting is the idea that is actually not a deep learning-centric problem at all; it's a problem that exists in all of machine learning. The key problem is how you can accurately define if your model is actually capturing your true data set or if it's just learning the subtle details that are only correlating to your data set.

01:03:08 - Said differently, let's say we want to build models that can learn representations from our training data that still generalize to brand new, unseen test points. The real goal here is that we want to teach our model something based on a lot of training data, but we don't want it to perform well only on the training data. We want it to do well when we deploy it into the real world, where it encounters things it has never seen during training.

01:03:34 - The concept of overfitting directly addresses this problem. Overfitting means that if your model is doing very well on your training data but very poorly on testing data, it is overfitting. It is overfitting to the training data that it saw. On the other hand, there's also underfitting. Underfitting occurs when the model does not fit the data enough, which means that it will achieve very similar performance on your testing distribution, but both are underperforming the actual capabilities of your system.

01:04:09 - Ideally, you want to end up somewhere in the middle—not too complex, where you're memorizing all of the nuances in your training data, but still able to perform well even with brand new data. You want to avoid underfitting as well.

# Regularization Techniques in Neural Networks

01:04:28 - To address this problem in neural networks and in machine learning in general, there are a few different techniques that you should be aware of, as you'll need to apply them as part of your solutions and software labs. The key concept here is called regularization. Regularization is a technique that you can introduce, and very simply, all regularization does is discourage your model from learning the nuances in your training data. This is critical for our models to be able to generalize—not just on training data, but really on the testing data, which is what we care about.

01:05:07 - The most popular regularization technique that is important for you to understand is a very simple idea called Dropout. Let's revisit this picture of a deep neural network that we've been seeing throughout the lecture. In Dropout, during training, we randomly set some of the activations (the outputs of every single neuron) to zero. We do this with some probability, say 50%. This means that we're going to take all of the activations in our neural network, and with a probability of 50%, before we pass that activation onto the next neuron, we will set it to zero and not pass on anything. Effectively, 50% of the neurons are going to be shut down or "killed" in a forward pass, and we will only forward pass information with the other 50% of our neurons.

01:06:02 - This idea is extremely powerful because it lowers the capacity of our neural network. Not only does it lower the capacity, but it does so dynamically. On the next iteration, we will pick a different 50% of neurons to drop out. Thus, the network is constantly forced to learn to build different pathways from input to output, and it cannot rely too extensively on any small part of the features present in the training data.

01:06:46 - The second regularization technique is called early stopping, which is model agnostic and can be applied to any type of model, as long as you have a testing set to work with. The idea here is that we have a formal mathematical definition of what it means to overfit. Overfitting occurs when our model starts to perform worse on our test set.

01:07:06 - If we plot the performance over the course of training, with the x-axis representing the training process, we can observe the performance on both the training set and the test set. Initially, both the training set and the test set performance decrease, which is excellent because it indicates that our model is improving. Eventually, however, the test loss plateaus and starts to increase, while the training loss continues to decrease. 

01:07:57 - The important point is that we care about the moment when the test loss starts to increase. This is the critical point where we need to stop training. After this point, we start to overfit on parts of the data, where our training accuracy becomes better than our testing accuracy. While our training accuracy improves, our testing accuracy begins to decline.

01:08:18 - On the other hand, on the left-hand side, we have the opposite problem. We have not fully utilized the capacity of our model, and the testing accuracy can still improve further. This is a very powerful idea, but it's actually extremely easy to implement in practice. All you really have to do is monitor the loss over the course of training, and you just have to pick the model where the testing accuracy starts to get worse.

# Summary and Transition to Next Lecture

01:08:46 - I'll conclude this lecture by summarizing three key points that we've covered in the class so far. This is a very packed class, so the entire week is going to be like this, and today is just the start. So far, we've learned the fundamental building blocks of neural networks, starting all the way from just one neuron, also called a perceptron. We learned that we can stack these systems on top of each other to create a hierarchical network and how we can mathematically optimize those types of systems.

01:09:16 - Finally, in the very last part of the class, we talked about techniques and tips for actually training and applying these systems in practice. Now, in the next lecture, we're going to hear from Ava on deep sequence modeling using RNNs, and also a really new and exciting algorithm and type of model called the Transformer, which is built off of the principle of attention. You're going to learn about it in the next class.

01:09:43 - But for now, let's take a brief pause, and we'll resume in about five minutes so we can switch speakers and Ava can start her presentation. Thank you.

