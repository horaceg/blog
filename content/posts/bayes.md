+++
title = "An introduction to bayesian inference"
date = 2024-04-18

[extra]
toc = true

[taxonomies]
categories = ["probability"]
tags = ["probability", "bayes", "inference"]
+++

I first learned about bayesian statistics in a job posting for an internship. At the time, I was about 20 years old and had never heard of it. The concept was so alien to me that I first read _bayesian **interference**_. I figured it had something to do with optics, such as the Michelson interferometer that I (still) fear.

Fast-forward a small decade and I now consider myself a bayesian. How did this happen ? What is bayesian statistics _really_ about ?

# Bayes' theorem

In the beginning of every probability book, you will see Bayes' thoerem proudly stated as a direct consequence of the definition of conditional probability and the product (chain) rule. I will argue here that it is not really helpful to think of bayes this way. If you wonder what Bayes' theorem is, you need to look it up because I won't talk about that here.

# A biased coin

When I was in high school, I remember that we saw in a book that if you toss a coin 3 times and it leads 3 heads, the probability that the next toss will lead head too is ... 0.5 (50%). I was a bit surprised by the result, because I overlooked the crucial fact that we knew the coin was unbiased, i.e. that it always leads to heads with a probability of 0.5.

However, if you don't know if the coin is biased or not, then your belief that the next coin toss is going to lead heads should surely be a little higher than tails. A relevant question you can ask yourself is: at what odds would you accept to bet on tails ? Assuming you want your expected gain to be positive of course.

# The reverend Thomas Bayes

| <img src="/images/thomas-bayes.png" align='center'> |
| ------------------------------------- |
| Only known portait of possibly Thomas Bayes. 18th century |

Born c. 1701, Thomas Bayes was a pioneer of statistics, and a minister of the Church. 

In 1763 was published _An Essay towards solving a Problem in the Doctrine of Chances_, 2 years after Bayes' death. 

In it, he states in the beginning the Bayes' theorem as we know it. [On wikipedia](https://en.wikipedia.org/wiki/An_Essay_towards_solving_a_Problem_in_the_Doctrine_of_Chances#Outline) we can read:
> it does not appear that Bayes emphasized or focused on this finding. Rather, he focused on the finding the solution to a much broader inferential problem: 
> "Given the number of times in which an unknown event has happened and failed [... Find] the chance that the probability of its happening in a single trial lies somewhere between any two degrees of probability that can be named."

