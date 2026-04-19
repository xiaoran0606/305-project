# Modeling Latent COVID-19 Dynamics with Socioeconomic and Demographic Covariate

## Data: 
- [Download data](https://docs.owid.io/projects/etl/api/covid/#download-data)
- [Data description](https://github.com/owid/covid-19-data/tree/master/public/data)

## Reference papers:
- [Forecasting COVID-19 new cases using deep learning methods](https://www.sciencedirect.com/science/article/pii/S0010482522001342)
- [Estimating the effects of non-pharmaceutical interventions on COVID-19 in Europe](https://www.nature.com/articles/s41586-020-2405-7)
- [Semi-Mechanistic Bayesian modeling of COVID-19 with Renewal Processes](https://rss.org.uk/RSS/media/File-library/Publications/JRSSA-Dec-2020-0025_Imperial-paper-preprint.pdf)
- [Estimating global, regional, and national daily and cumulative infections with SARS-CoV-2 through Nov 14, 2021: a statistical analysis](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(22)00484-6/fulltext)
    - Justify how the number of deaths attributed to COVID-19 can be corrected using the excess mortality rate.
    - Basic idea: excess mortality rate -> counterfactual excess mortality rate due to COVID-19 -> corrected total COVID-19 deaths.
    - Note: estimating the counterfactual excess mortality rate requires referring to another [statistical prediction model for excess mortality](https://www.thelancet.com/article/S0140-6736(21)02796-3/fulltext). Here, we adopt [a cruder approach as an approximation](https://pubmed.ncbi.nlm.nih.gov/34190045/) of that idea.

## Tentative modeling ideas:
- State Space Model
- Gaussian Process
- Hierarchical Model (country? continent?)