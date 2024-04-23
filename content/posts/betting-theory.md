+++
title = "Mathematics of bookmaking"
date = 2021-09-06

[extra]
toc = true

[taxonomies]
categories = ["probability"]
tags = ["math", "probability", "bets", "theory", "framework"]
+++

# Introduction

Have you ever placed a bet ? Wondered what the odds actually mean ?

It's all linked to probability.

# Setup

Let us place focus on a betting situation: a bookie offers odds on an event $A$, and on its complementary $\overline{A}$. There can be multiple binary events at the same time (or close). We will then extend to multiple outcomes.

In our case, we consider real-life processes, which are physical, and their outcomes. We regularly take the example of sports gambling, where processes are fixtures, which generate various outcomes.

For a given bet, we have (decimal) _odds_ $o_A$ such that if you place a bet with _stake_ $s$ on event $A$, you lose $s$ in all cases, and if you win, you get $so_A$ in addition. So your total net gain is $s(o_A - 1)$ in case of a win, and $-s$ otherwise.

Denote 

* $\pi\_{A} := \frac{1}{o_A}$ the inverse odds. This quantity is usually _not_ a probability.

Define the net unit gain:

* $G^o_A := o_A \mathbf{1}_A - 1$

$sG^o_A = s(o_A \mathbf{1}_A - 1)$ is then the net gain of placing a bet on event $A$ with stake s and inverse odds $\pi_A$.

Let us also define the _natural_ bet, that is the bet on $A$ with a stake of $\pi_A$:

* $F^{\pi}_A := \pi_A G^o_A = \mathbf{1}_A - \pi_A$

We will see that this quantity is useful to stake the bets in order to combine them. Intuitively, it makes sense to stake bets with low odds with large money since you are (supposedly) likely to win, and bets with high odds with fewer money since you are (supposedly) unlikely to win.

Define $p$ a probability on the universe $\Omega$, and $p_A$ the probability of $A \subset \Omega$. This might be _the true_ probability law on sports events, or your belief of it, or someone else's belief. All we know it that it is a coherent belief since $p$ satisfies the axioms of probability. 

A bet is _sure_ when its gain is deterministic, i.e. there is not randomness involved. 

When there is no possible confusion, we write $G_A \cong G^o_A$ and $F_A \cong F^{\pi}_A$

# Coherent odds

<!-- [Todo: Quelle est la relation de $\pi$ avec $\hat{p}$ ? qqc à propos de la continuité absolue de $\pi$ par rapport à $p$ et la dérivée de Radon-Nikodym? Voir aussi] -->
<!-- [Todo [Girsanov](https://en.wikipedia.org/wiki/Girsanov_theorem)] -->


## Coherence

We say that the odds are _coherent_ for a given bookie, or that a _book_ is coherent, if we have the following:

* $\pi\_{A \cup B} = \pi_A + \pi_B$ when $A \cap B = \emptyset$

It is another way to say that $\pi$ is additive.

Then we can deduce other rules:

* $\pi\_{A \cup B} = \pi_A + \pi_B - \pi\_{A \cap B}$

* $\forall A, \pi\_{\Omega} = \pi_A + \pi\_{\overline{A}}$

This last equality shows that the **booksum** is universal. This also allows us to prove that 

* $\pi\_{\emptyset} = 0$

Which means that $\pi$ is a measure, and which is finite. 

<!-- 
## Additional assumptions

Assume that $\pi$ is usually absolutely continuous with respect to $p$, that is every event with zero probability doesn't have any odds (or odds of $+\infty$, theoretically). 

[If not, apply the Lebesgue decomposition $\pi = \pi^1 + \pi^2$ where $\pi^1$ is absolutely continuous with respect ot $p$ and $\pi^2$ is orthogonal to $p$; then let $\pi \leftarrow \pi^1$].

Then we can compute the Radon-Nikodym derivate:

$h := \frac{d\pi}{dp}$  thus $\pi_A = \int_A h dp = \mathbb{E}[h\mathbf{1}_A]$. 
 -->

Warning: $\pi\_{\overline{A}} \neq 1 - \pi_A$, rather $\pi\_{\overline{A}} = \pi\_{\Omega} - \pi_A$. We will see that afterwards, this allows the bookie to make a profit.

If we apply the right transformation to $\pi$, we might get (approximately) the **implied probability** law $\hat{p}$ estimated by the bookie, and his margin on each bet. 

Note that the popular saying that $\pi \simeq \hat{p}$ is false, since those do not sum to 1. To go further see the [implied probability](#implied-probability) section.

## Product

We say that  $A {\perp \\!\\!\\! \perp} B$ (for $\pi$) if we have $\pi\_{A \cap B} = \pi_A \pi_B$.

* For $A {\perp \\!\\!\\! \perp} B$, $G\_{A \cap B} + 1 = (G_A + 1)(G_B + 1)$

i.e.

* $F\_{A \cap B} = (F_A + \pi_A)(F_B + \pi_B) - \pi_A \pi_B$ 

## Additivity

Let $A$, $B$ such that $A \cap B = \emptyset$. We have : 

* $G\_{A \cup B} = \frac{\pi_A}{\pi_A + \pi\_{\overline{A}}} G_A + \frac{\pi_B}{\pi_A + \pi\_{\overline{A}}} G_B$.

i.e.

* $F\_{A \cup B} = F_A + F_B$.

In the general case, when $A \cap B \neq \emptyset$, we have:

* $F_A + F_B = F\_{A \cap B} + F\_{A \cup B}$

## The sure universal bet

When we apply the additivity formula for $B \leftarrow \overline{A}$, we get:

* $G\_{\Omega} = \frac{\pi_A}{\pi_A + \pi\_{\overline{A}}} G_A + \frac{\pi\_{\overline{A}}}{\pi_A + \pi\_{\overline{A}}} G\_{\overline{A}} = \frac{1}{\pi_A + \pi\_{\overline{A}}} - 1$

i.e.

* $F\_{\Omega} = F_A + F\_{\overline{A}} = 1 - (\pi_A + \pi\_{\overline{A}})$

Unsurprinsingly, this is a sure bet. It is the opposite of the _margin_ of the bookie. That is the gain of the unit bet on the sure event $\Omega$, placing a bet on both $A$ and $\overline{A}$ with the proper stakes ratio.

By identifying $G\_{\Omega} = \mathbf{1\_{\Omega}}\frac{1}{\pi\_{\Omega}} - 1$, we recognize that:

* $\pi\_{\Omega} = \pi_A + \pi\_{\overline{A}}$

This quantity $\pi_A + \pi\_{\overline{A}}$ is the **booksum** in general (for a given event $A$).

Let us write what is now trivial:

* $G\_{\Omega} = \frac{1 - \pi\_{\Omega}}{\pi\_{\Omega}} = \frac{1}{\pi\_{\Omega}} - 1 = o\_{\Omega} - 1$

* $F\_{\Omega} = 1 - \pi\_{\Omega}$


## Transition

Note that we always have $F_A + F\_{\overline{A}} = 1 - (\pi_A + \pi\_{\overline{A}})$, even if the odds are uncoherent. However, if the odds are not coherent, we can't say that this quantity is indeed $F\_{\Omega}$. For instance, the bookie could theoretically offer other odds for the event $\Omega$, leading to $\pi\_{\Omega} \neq \pi_A + \pi\_{\overline{A}}$. However, would they really offer such a bet ?

In the following sections, we do not assume coherent odds. However, we use the notations 

* $\hat{\pi}\_{\Omega} := \pi_A + \pi\_{\overline{A}}$

* $\hat{G}\_{\Omega} := \frac{1}{\pi_A + \pi\_{\overline{A}}} - 1$

* $\hat{F}\_{\Omega} = 1 - (\pi_A + \pi\_{\overline{A}})$

# Fairness

A _fair_ bet implies $\mathbb{E}_p[F_A] = 0$. Since $\mathbb{E}_p[F_A] = p_A - \pi_A$, we then have:

* $p_A = \pi_A = \frac{1}{o_A}$. 

In this case, the odds are coherent since $\pi = p$ is a probability measure. We have $\frac{1}{o_A} + \frac{1}{o\_{\overline{A}}} = 1$, that is $o\_{\overline{A}} = \frac{o_A}{o_A - 1}$.

We also have the following: 

<!-- * $o_A + o\_{\overline{A}} = o_A o\_{\overline{A}}$ -->
<!-- * $mG_A + nG\_{\overline{A}} = 0 \iff n = m(o_A - 1)$ -->
* $\pi\_{\Omega} = 1$
 
* $G\_{\Omega} = F\_{\Omega} = 0$

This last formula is crucial to understand fairness. It is useful to hedge your bets with the right amount, since e.g. $-G_A = \frac{o_A}{o\_{\overline{A}}} G\_{\overline{A}}$. More on this at [Negative bets](#negative-bets).

Note that even if bookkeepers were unwilling to make a profit, they don't really know the true probability (is there such a thing?). So the best they can do is to offer odds that are fair for a probability measure $\hat{p}$, that is the reflection of their honest and coherent beliefs. This is equivalent to $\pi = \hat{p}$, and thus to the two following statements:

* $\pi$ is coherent (additive), e.g. $\pi$ is a (finite) measure

* $\pi\_{\Omega} = 1$

# Arbitrage

Let's compute the original hedging quantity that sums to zero in a fair betting situation. Recall that $\hat{G}\_{\Omega}$ is the gain of placing a bet on $A$ combined with a bet on $\overline{A}$ with the proper stakes. In an unfair (real) situation, this does not sum to zero.

If the booksum $\hat{\pi}\_{\Omega} < 1$, then $\hat{G}\_{\Omega} > 0$: this means that one can bet on both outcomes and make a sure profit, according to any outcome. This is called an _arbitrage_ (or _arb_).

Thus, a bookie needs to set the booksum $\hat{\pi}\_{\Omega} \geq 1$, otherwise this will lead to quick bankruptcy: everyone _in the know_ will bet on both outcomes (with the proper stakes ratio).

Moreover, the higher the booksum, the higher the profits for bookies since $\hat{F}\_{\Omega} = 1 - \pi\_{\Omega}$. Recall that your loss is the bookie's gain.

However, since maximizing the booksum means minimizing the average odds, informed bettors will likely place their bets with another bookie. Thus, for the bookie, it is all about maximizing the booksum while still attracting customers, keeping in mind that competitors do the same.

# Value betting

Let us now consider this _smart bookie_ configuration: $\pi\_{\Omega} > 1$. Is there still any opportunity to make money as a bettor (and lose money as a bookie)? Well, _probably_. Recall the expected gains for a given event:

$\mathbb{E}_p[F_A] = p_A - \pi_A$

This means that if you trust your belief $p_A$, and that it is higher than the one the bookie offers you:

* $p_A > \pi_A = ( 1 + \epsilon_A) \hat{p}_A$ with $\epsilon_A$ the margin

then you should bet. In practice, you should probably look for differences that are wide enough for you to bet.

However, in this case there is no sure bet, and hence no sure profit : there is only an estimate of the expectation. Hence, this is much riskier than an arbitrage. In order to make a profit based on this, you need to bet on many opportunities where you find a positive expectation, so that on average your gain should be positive. If you have a robust estimation method, then you will make a profit if you size your bets correctly (see e.g. the Kelly criterion). On the other hand, if you your estimates are often off compared to the bookies', then you will eventually go bankrupt (see the gambler's ruin).

# Negative bets

_Negative_ bets are simply when you take the role of the bookie: you offer someone else a bet. Then, their gain is exactly your loss, and vice-versa. Hence, placing a negative bet is simply the opposite of the positive bet that the person takes. But what if you cannot offer someone else a bet at the price you buy it? It is hard to find either a bookie that allows you to take his place, or an exchange that does not take commissions.

Well, we have to resort to $\hat{F}\_{\Omega}$. Recall that, in the general case:

$\hat{F}\_{\Omega} = F_A + F\_{\overline{A}} = 1 - \hat{\pi}\_{\Omega}$.

If the __fair situation__, $\hat{F}\_{\Omega} = 0$. Thus, we can simply place the opposite bet with the proper stake: 

* $- F_A = F\_{\overline{A}}$

However, in the __unfair situation__, considering a smart bookie, we have:

* $F\_{\overline{A}} = - F_A - \epsilon_A$

where $\epsilon_A = \hat{\pi}\_{\Omega} - 1 > 0$. This is the bookie's margin: you cannot hedge yourself at net zero cost. You must pay the price to the bookie. You can clearly see that again by placing a sure net negative bet:

* $F_A + F\_{\overline{A}} = - \epsilon_A < 0$

Note that if the bookie is not smart, or that you are dealing with several bookies at the same time, treating them as one bookie offering the highest odds for any event, you might see an opportunity where $\hat{F}\_{\Omega} > 0$, that is $\epsilon_A < 0$. This quantity becomes you sure net gain in this arbitrage case.

# Incoherent odds

## Incompatible events

Let's consider the case when the odds are incoherent. In the case of $A \cap B = \emptyset$, if you chose to bet both on $A$ and $B$ separately, then you can actually (implicitly) bet on $A \cup B$ with the proper stakes proportion. We assume that the bookie offers (incoherent) odds for $A$, $B$, $A \cup B$ and $\overline{A \cup B} = \overline{A} \cap \overline{B}$. This situation never arises in the scope of binary outcomes sports but rather typically in football (soccer): $\{A, B, \overline{A \cup B}\} \cong \\{Home, Away, Draw\\}$

We introduce proper notation:

* $C := A \cup B$

We want to compare $F_A + F_B$ and $F_C$.

## One side

We compute the theoretical sum of a positive and a negative bet: 

$F_A + F_B - F_C = \pi_C - (\pi_A + \pi_B)$

If the "coherent" version of the inverse odds for $C$ are low enough, there is an opportunity. One has to bet with the inverse formula:

$F\_{\overline{C}} = - F_C + (\pi_C + \pi\_{\overline{C}} - 1)$

Consider the betting opportunity:

* $\hat{F}\_{\Omega} := F_A + F_B + F\_{\overline{C}} = (\pi_C - \pi_A - \pi_B )- (\pi_C + \pi\_{\overline{C}} - 1) = 1 - (\pi_A + \pi_B + \pi\_{\overline{C}})$

We have a sure (triple) bet. We will make a profit if and only if this quantity is positive, i. e.:

* $\hat{\pi}\_{\Omega} := \pi_A + \pi_B + \pi\_{\overline{C}} < 1$

This is exactly the same arbitrage formula as before, but with three outcomes.

## The other side

This time we target $F_C - F_A - F_B$. In this case, we place the following bets:

$F\_{\overline{A}} = - F_A - (\pi_A + \pi\_{\overline{A}} - 1)$

We use exactly the same formula for $B$, and we get a betting opportunity:

$F\_{\overline{A}} + F\_{\overline{B}} + F_C = 2 - (\pi\_{\overline{A}} + \pi\_{\overline{B}} + \pi\_C)$

## General case

This time, we do not assume $A \cap B = \emptyset$.

We leverage the following: 

* $1\_{A \cap B} = 1_A + 1_B - 1\_{A \cup B}$ i.e. $1_D = 1_A + 1_B - 1_C$

* $1\_{A \cap B} = 1_A 1_B$.

Let $D := A \cap B$.

We essentially want to compare $F_D$ and $F_A + F_B - F_C$.

Then, consider the following pseudo-bet with the pseudo-odds $\pi_A + \pi_B - \pi_C$:

$F_A + F_B - F_C = \mathbf{1}_A + \mathbf{1}_B - \mathbf{1}_C - (\pi_A + \pi_B - \pi_C)$

Now we need to compare this to 

$F_D = \mathbf{1}_A + \mathbf{1}_B - \mathbf{1}_C - \pi_D$

### One side

First, we target $F_A + F_B - F_C - F_D$. In order to actually place an approximation of this bet, we need to reverse the two negative bets $-F_C$ and $-F_D$.

First we have:

$F\_{\overline{C}} = - \mathbf{1}_C + \pi_C + 1 - (\pi_C + \pi\_{\overline{C}}) = - \mathbf{1}_C + 1 - \pi\_{\overline{C}}$


With the same reasoning, we have use $F\_{\overline{D}} = - F\_{D} - (\pi_D + \pi\_{\overline{D}} - 1) = - (\mathbf{1}_A + \mathbf{1}_B - \mathbf{1}_C) + 1 - \pi\_{\overline{D}}$.

We get the sure bet combinations:

$\hat{F}\_{\epsilon_1} := F_A + F_B + F\_{\overline{C}} + F\_{\overline{D}} = 2 - (\pi_A + \pi_B - \pi\_{\overline{C}} - \pi\_{\overline{D}})$


### The other side

Consider the opposite quantity:

$F_D + F_C - F_A - F_B$

We need to reverse the bets on A and B to get a sure bet:

$\hat{F}\_{\epsilon_2} := F\_{\overline{A}} + F\_{\overline{B}} + F_C + F_D = 2 - (\pi_C + \pi_D - \pi\_{\overline{A}} - \pi\_{\overline{B}})$


Provided we can place bets on $A$, $B$, $\overline{C} = \overline{A \cup B} = \overline{A} \cap \overline{B}$ and $\overline{D} = \overline{A \cap B} = \overline{A} \cup \overline{B}$, we need to check the two above-mentioned quantities. However, it is usually not possible to compose bets in such a fashion: the bookies does not offer any bet on $\overline{A} \cup \overline{B}$.

<!-- 
### Compatible events

[WIP]

If we don't necessarily have $A \cap B \neq \emptyset$, then we need to set the pseudo-bet:

## Product

[WIP]

Consider $A {\perp \\!\\!\\! \perp} B$. We cannot simply bet on the intersection of two bets, we can only bet on the union. We need to use set theory: $A \cap B = \overline{\overline{A} \cup \overline{B}}$. Now, let use build a composite bet on this event.

Let :

* $C := A \cap B$.

* $\hat{\pi}_C := \pi_A \pi_B$

Note that $\overline{C} = \overline{A} \cup \overline{B}$.

* $\pi_C G_C = - \pi\_{\overline{C}} G\_{\overline{C}} - \epsilon$

Recall the coherent case, and compute the difference:

* $\pi_C G_C - \pi_A \pi_B [(G_A + 1)(G_B + 1) - 1] = \pi_A \pi_B - \pi_C$ -->


# Thinking like a bookie
<!-- 
Let us consider the case of an obviously _mispriced_ event: imagine a game of basketball, where there is no draw. Assume that the underdog has lower odds than the favourite: if $A := [ \text{The favourite wins} ]$, $o_A > o\_{\overline{A}}$ thus $\pi_A < \pi\_{\overline{A}}$. The "true" probabilities are such that $p_A > p(\overline{A})$. Then you just need to bet on the favourite:  -->

If we think like a bookie, here are our tasks in order to thrive:

* Estimate the true probabilities as precisely as possible
* Set the odds lower than the estimated ones (inverse odds are thus higher than estimated probabilities)
* Attract bettors by offering high enough odds
* Minimize the risk

<!-- This could be expressed as:
* $\miF\_{\pi_A, \pi\_{\overline{A}}} \hat{p}_A - \pi_A + \hat{p}\_{\overline{A}} - \pi\_{\overline{A}}$
* $\max\_{\pi_A, \pi\_{\overline{A}}}|\pi_A + \pi\_{\overline{A}}|$ -->

As a bookie, you want to minimize the _risk_, i.e. the largest sum of money you could potentially lose. Consider for example that a large number of people do bet on an unlikely outcome, with high odds, and this event realizes. In order to avoid this, the bookie can intentionally skew the odds while realizing that many people (or few people with large sums of money) are betting on this outcome. This can also mean that there is a value opportunity for bettors, i.e. that the odds are not set properly. In this example, the bookie can monitor the volume of bets, while computing the risk, and decrease it when it is too high by decreasing the corresponding odds and increasing the odds of opposite events, in order to attract bettors. He can also place bets at another bookie in order to hedge, although this is probably not favoured in a context of fierce competition.

<!-- 
The above discussion is valid for bookies, but also for players that want to play seriously, that is invest large sums of money in the market. Even if you can't set the odds, eventually it is the same: you have your estimates of the true probability, another player has his, and you confront them. Bookies that set the odds need to be careful to balance their books in order not to create a _dutch book_ against them. -->

# Implied probability

We want to know the _implied probabilities_ of the odds, the probability estimated by the bookie. We actually need to reverse engineer the recipe that they use in order to maximize profit based on their estimate of the true probabilities, in order to recover them. This is basically an inverse problem.

Denote:

* $\pi_A = (1 + \epsilon_A) \hat{p}_A$

with $\hat{p}_A$ the bookie's (coherent) belief, $\epsilon_A$ the bookie's margin on event A. The bookie certainly hopes that $\epsilon_A > 0$, and tries to do that.

## Basic normalization

An easy technique to get a probability from consistent (and unfair) odds is to normalize the inverse odds:

* $\hat{p} := \frac{1}{\hat{\pi}\_{\Omega}}\pi$

This assumes that the $\forall A, \epsilon_A = \hat{\pi}\_{\Omega} - 1$: the bookies' margin is uniform on all outcomes, which is probably false in real situations. However, this provides a first guess that is easy to compute.

If we assume that $\pi$ is coherent, it is immediate that we have $\hat{p}\_{\Omega} = 1$ and thus $\hat{p}$ is a probability measure.

## Shin's model

See Shin 1991, Strumbelj 2014 for further analysis.

<!-- # Sizing bets

## Kelly criterion

[Todo] -->

# Multiple odds

## Single event

Observe that for any given event, $\\{F^{\pi}\ | \pi \ge 0 \\}$ is a convex set:

$\alpha F^{\pi_1} + (1 - \alpha) F^{\pi_2} = F^{\alpha \pi_1 + (1 - \alpha)\pi_2}$.

This gives us a useful formula to combine gains, betting on mutiple odds at once:

$x F^{\pi_1}_A + y F^{\pi_2} = (x + y) (\frac{x}{x + y}F^{\pi_1} + \frac{y}{x+y}F^{\pi_2}) = (x + y) F^{\alpha \pi_1 + (1 - \alpha) \pi_2}$

where $\alpha = \frac{x}{x+y}$.

In practice, you can also use $G$ with the odds:

$x G^{o_1} + y G^{o_2} = (x + y) G^{\alpha o_1 + (1 - \alpha) o_2}$

If we have access to multiple odds at a given time, we have virtually access to all the odds in between. Why would you bet in-between though, since you should take the highest odds?

Also, this trick might be useful to _reduce_ a position to one equivalent bet, or _expand_ a single bet into multiple ones. This might also be useful to test a risk assessment system (with e.g. a property-based testing framework).

A more realistic situation arises when we place multiple bets on the same event at different times, while odds have been varying (in-play betting for example). If you did spot value at time $t$ and the odds move in your favour, then you should still bet at time $t + \delta t$. In order to simplify you position, you can virtuallly reduce your two bets to one equivalent bet at odds in-between.

<!-- # Odds varying through time

[todo]

$\pi_A^t$ represents the inverse-odds through time. -->

# Appendix

## Measure theory

* $\mu (A) = \int_A \mathbf{1}_Ad\mu$

* $1\_{A \cap B} = 1_A 1_B$

* $1\_{A \cup B} = 1_A + 1_B - 1\_{A \cap B}$

## Probability theory

A probablity $p$ defined on any $A \subset \Omega$ is a finite measure that satisfies the property $p\_{\Omega} = 1$.

* $E_p[1_A] = p_A$

* We say $A {\perp \\!\\!\\! \perp} B$ for $p$ iff $p\_{A \cap B} = p_A p_B$ i.e. $E_p[\mathbf{1}_A \mathbf{1}_B] = \mathbb{E}_p[\mathbf{1}_A] \mathbb{E}_p[\mathbf{1}_B]$ 

* More generally, we say $X {\perp \\!\\!\\! \perp} Y$ for $p$ iff $E_p[XY] = E_p[X] E_p[Y]$
