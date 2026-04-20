# Modeling Latent COVID-19 Dynamics with Socioeconomic and Demographic Covariate

## Data: 
- [Download data](https://docs.owid.io/projects/etl/api/covid/#download-data)
- [Data description](https://github.com/owid/covid-19-data/tree/master/public/data)

## Reference papers:
- [Forecasting COVID-19 new cases using deep learning methods](https://www.sciencedirect.com/science/article/pii/S0010482522001342)
- [Estimating the effects of non-pharmaceutical interventions on COVID-19 in Europe](https://www.nature.com/articles/s41586-020-2405-7)[and the supplementary material](https://www.nature.com/articles/s41586-020-2405-7#MOESM1)
    - Modeling the number of true infections using a discrete renewal process
- [Semi-Mechanistic Bayesian modeling of COVID-19 with Renewal Processes](https://rss.org.uk/RSS/media/File-library/Publications/JRSSA-Dec-2020-0025_Imperial-paper-preprint.pdf)
- [A new framework and software to estimate time-varying reproduction numbers during epidemics](https://pubmed.ncbi.nlm.nih.gov/24043437/)
    - Renewal process
- [Estimating global, regional, and national daily and cumulative infections with SARS-CoV-2 through Nov 14, 2021: a statistical analysis](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(22)00484-6/fulltext)
    - Justify how the number of deaths attributed to COVID-19 can be corrected using the excess mortality rate.
    - Basic idea: excess mortality rate -> counterfactual excess mortality rate due to COVID-19 -> corrected total COVID-19 deaths.
    - Note: estimating the counterfactual excess mortality rate requires referring to another [statistical prediction model for excess mortality](https://www.thelancet.com/article/S0140-6736(21)02796-3/fulltext). Here, we adopt [a cruder approach as an approximation](https://pubmed.ncbi.nlm.nih.gov/341 90045/) of that idea.
- [Estimating global, regional, and national daily and cumulative infections with SARS-CoV-2 through Nov 14, 2021: a statistical analysis](https://pubmed.ncbi.nlm.nih.gov/35405084/)
    - Estimate of IDR

## Tentative modeling ideas:
We build a **hierarchical state space model** where the true number of infections $$z_{t,c}$$ (where t denotes time and c denotes country) is a hidden quantity modelled as a **discrete renewal process** (conventional approach in epidemiology) that drives two noisy observations (assumed to be the mean of the **Negative Binomial distribution**) — reported cases and reported deaths. Reported cases are biased by testing capacity, which we correct using ⁠`tests_per_case` as a proxy for the infection detection ratio (IDR). Reported deaths are biased by attribution errors, which we correct using ⁠`excess_mortality`⁠. 


The reproduction number $$R_{t,c}$$ (a key component of the discrete renewal process)​ evolves over time as a function of country-specific covariates, such as policy stringency and vaccination coverage. Country-level parameters are linked through continent-level priors, constituting the hierarchical structure of the model.