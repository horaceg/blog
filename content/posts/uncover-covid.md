+++
title = "Bayesian epidemiology"
date = 2021-09-09

[taxonomies]
categories = ["probabilistic programming"]
tags = ["machine learning", "probability", "bayesian", "epidemiology", "kaggle", "ppl", "programming", "probabilistic programming"]
+++

In March 2020, lockdown measures were implemented all over the world due to the spread of Covid-19. I was at home, a bit bored and I was looking for something to do.

__TL;DR__: 
- [Kaggle submission](https://www.kaggle.com/achyrogue/impact-of-interventions-through-mobility-data)
- Github [repo](https://github.com/horaceg/uncover)

# Mobility datasets

[Google](https://www.google.com/covid19/mobility/) and [Apple](https://covid19.apple.com/mobility) published datasets related to (anonymized) activity of smartphones users. They called this _mobility_ data, although this doesn't show trajectories but rather the frequentation of some categories of places, such as grocery stores, residential or workplaces.

At first, there were no csv files, only pdf graphs that did not show raw data. I found a reddit post that extracted some of the data from the reports (for US regions), and I extended it to make it work with worlwide data. I then published it on a [small website](https://covid19-analysis.netlify.app/posts/mobility/) I made for the occasion.

The code to do the download + parsing + extraction is available on a [github repo](https://github.com/horaceg/covid19-analysis/tree/master/mobility)

# Uncover challenge

The [Uncover challenge](https://www.kaggle.com/roche-data-science-coalition/uncover) appeared on Kaggle a few days afterwards. From the description:

> The Roche Data Science Coalition (RDSC) is requesting the collaborative effort of the AI community to fight COVID-19. This challenge presents a curated collection of datasets from 20 global sources and asks you to model solutions to key questions that were developed and evaluated by a global frontline of healthcare providers, hospitals, suppliers, and policy makers. 

The task that immediately caught my attention:

> How is the implementation of existing strategies affecting the rates of COVID-19 infection?

I had a (not so secret) weapon: mobility data.

# Exploration

First I did some exploration and dataviz leveraging the [ACAPS](https://www.acaps.org/covid-19-government-measures-dataset) dataset, Google mobility data and epidemiology data (number of cases, deaths etc.).

I exported a jupyter notebook to html, and since I leveraged [Altair](https://altair-viz.github.io/) based on [Vega-lite](https://vega.github.io/vega-lite/), the data is embedded in the js so it works without a server. 

<!-- Try to interact with the plots in the [exported notebook](/measures_and_rate_explo.html). -->


# Compartmental models

Then I found this [Kaggle notebook](https://www.kaggle.com/anjum48/seir-hcd-model) by @anjum48 (_Datasaurus_), published in the Covid forecasting competition (another Kaggle competition). I like the scientific approach: based on well-known [Compartmental models](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model) in epidemiology, he extends it to fit the Covid-19 situation.


# Recruiting the team

After this preliminary work, I reach out to two friends for a collaboration: a data scientist and a quant analyst. They were also motivated by the project ! So we went to work on the challenge with this special forces team.

# Probabilistic programming

I had heard about bayesian inference before, but did not know how it worked. We decided to explore this path. As a first step, we chose a framework. We went for the young an promising [Numpyro](http://num.pyro.ai/en/stable/getting_started.html), based on [JAX](https://github.com/google/jax/) and made by some of the (very skilled) [Pyro](https://pyro.ai/) authors. This is also how I discovered JAX.

An example in particular seemed to fit our problem: a prey-predator differential equations [model](http://num.pyro.ai/en/stable/examples/ode.html).

Imperial College in London had already started to publish a series of papers called the [Covid-19 reports](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/covid-19-reports/). Report 13 was particularly relevant for our challenge, and can be found [here](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-13-europe-npi-impact/).

# Collaborating on a Jupyter notebook

Kaggle required a notebook submission. Collaboration on notebook is particularly challenging: indeed, the JSON format that mixes code and data is not git-friendly. So, I developed a small ipython magic command that allows us to modify code inside a notebook cell, and when executed it saves it into a specific file. The notebook wasn't version-controlled, but every cell snippet was. It wasn't ideal, since the imports were not included in every cell.

# The submission

_In fine_, we implemented bayesian Covid-19 models that perform:
- An estimation of the effective reproduction number Rt as a function of mobility data and reported deaths
- An estimation of mobility as a function of non-pharmaceutical interventions

See the [kaggle submission](https://www.kaggle.com/achyrogue/impact-of-interventions-through-mobility-data) for the full analysis.

The Roche team accepted our submission, so we won the challenge for this particular task.
