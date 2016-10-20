---
layout: post
title: Project 4 - Fantastic 4 Consultancy - Predicting Data Scientist Salaries via Web Scraping
date: 2016-10-19 02:30:00
summary: Overview and description of my introductory Data Science Immersive project
categories: project dsi python eda web scraping LogisticRegression salary glassdoor
---

Week 4 Project Overview
-----------------------

This week's assignment was a group effort with 3 other classmates, Amish, Jesse and Kristen.

The project was designed to determine the factors which contribute to salaries of Data Scientists being either above the median of all salaries or below. Utilizing web scraping techniques and Logistic Regression (Classification).

Specifically the project required us to:

-	identify public web sources for job listings with Salary data.
-	obtain, via Web Scraping, enough data and features to perform statistical tests
-	clean the data, as necessary
-	develop a reproducible method supported by sound practices for making statements about what features are most influential in position salaries
-	utilize visual and statistical tools to present our recommendations
-	use new statistical tools and functions such as KNeighborsClassifier, GridSearchCV, selenium/phantomJS web tools, LogisticRegression

[Project 4 Jupyter Notebook - Group Submission](https://github.com/jpfreeley/GA-DSI/blob/master/DSI_IMAGE/curriculum/week-04/4.1-lab-webscraping/scraping-project-4-starter_JPF-Jesse.ipynb)

**Note:** The methodology below is drawn from the "Data Science Workflow" document provided by General Assembly

IDENTIFY THE PROBLEM
====================

*We've been asked to determine which parts of a job description are most influential in determining the salary of a Data Science "flavored" position. Because it is known that several job titles overlap in the Data Science industry, we must be wary of these subtleties*

**Risks and Assumptions**

It was clear that obtaining a large dataset which contained salary information was going to be a challenge. By 1st scraping the indeed.com website, we got a sense that about 4% of job descriptions contained some indicator of salary.

We were also at the mercy of the indeed.com search algorithm to present us relevant search results. We also needed to assume that there were no gross typographical errors. We dd not have significant outliers.

To supplement the indeed.com data, we also scraped historical salary information from glassdoor.com. This allowed us a baseline of Location and Title information along with salaries.

### ACQUIRE THE DATA
We were successful in scraping job postings from indeed.com. We scraped up to 1000 listings for each of 39 cities. In the end we compiled nearly 27,000 listings, about 1000 of which contained salary data. With the following data for each:

Location of Job - string
Title of Job - string
Summary of Job description - string
Job Posting Company - string
JKID - unique indeed.com job identifier - string
Number of "stars" for that Company - float 0-5 (0.5 increments)
Number of reviews for the company - integer
How many days ago the posting was created - float
Date that the posting was scraped - datetime
City that was used for the search - string

From Glassdoor we obtained the following data for nearly 1400 records:

salary
Location
title
company

Lastly, we obtained a Cost of Living index for cities in our scraped dataset from expatistan.com.

### PARSE THE DATA
Much of our efforts was spent cleaning the various columns. We had functions for cleaning the reviews, the salary, the post_date, the title, and various other fields we felt needed standardization.

One process that included a manual inspection and tagging of the data was the "binning" of the title data into 3 separate bins names ENTRY-LEVEL, MID-LEVEL, SENIOR-LEVEL experience level. We wrote code to programmatically bin the data, however the initial work to determine which title keywords should be assigned to which bin was a process that required some manual decisions.

This binning step was very important because these bins were directly used as features in our model as well as a way to join the glassdoor dataset. Decisions to assign a particular job listing as ENTRY/MID/SENIOR were somewhat subjective and could have significantly influenced our results. It would have perhaps been better to use countvectorizer to tokenize the titles and assign weights such as word occurrence. Further work would be required to investigate this more fully.

Likewise, glassdoor data was prepared, and titles binned by experience level as above.

### MINE THE DATA

Our overall goal was to attempt to make recommendations regarding which parts of a job listing might be used to determine if the positions salary would be above or below the median. I think we may have missed the mark slightly in this task. Because of the shortcomings of the glassdoor dataset, we were ultimately only able to use City and the binned Experience Level as the features of our model.

Due to this constrained feature-set, the analysis which was performed was more academic than enlightening. What we did was use the glassdoor data to train our model and compared that to the test set of the indeed data. The results and predictions below are a results of this train-test methodology.

#############

In conclusion, we can surmise that our model can, with reasonable accuracy, make a prediction about whether a salary would be above or below the median salary in the genre of "Data Scientist" jobs.

### REFINE THE DATA

We have already mentioned above that we chose to eliminate outlying stores based on a sales threshold of $100K. We then aggregated our targets of total sales and total volume.

In addition to these manipulations, we also aggregated total number of stores per zip code.

Two other fields were computed at this point which will be used during our recommendation phase.

- **Sales Dollars per Liter** - was derived to denote whether a zip code should be considered an expensive, average or inexpensive liquor market.
- **Stores per Sq Kilometer** - was derived to give a sense of whether the market was oversaturated with Liquor stores or whether there was a potential for more supply.

### BUILD A DATA MODEL

With our targets ready and our demographic features data in hand, we set out to build 2 separate models. For each we used LassoCV (with our cv argument set at 5) in order to assist in elimination of features that the algorithm deemed as unimportant.

```
X = model_df[features]
y_sales = model_df['Zip Sales - Total']
lasso = linear_model.LassoCV(cv=5)
model_sales = lasso.fit(X,y_sales)
print 'r-squared: {}'.format(model_sales.score(X,y_sales))
print 'alpha applied: {}'.format(model_sales.alpha_)
```

```
X = model_df[features]
y_volume = model_df['Zip Volume - Total']
lasso = linear_model.LassoCV(cv=5)
model_volume = lasso.fit(X,y_volume)
print 'r-squared: {}'.format(model_volume.score(X,y_volume))
print 'alpha applied: {}'.format(model_volume.alpha_)
```

Sales  | Volume
--|--
![](/images/project-03/predictedVactualSales.png)  |  ![](/images/project-03/predictedVactualVolume.png)
- an r-squared value of: 0.816989065682 | - an r-squared value of: r-squared: 0.803698824346
- an alpha applied value of: 3512.74084712 | - an alpha applied value of: 237.245609349


It's interesting to note that some of the features that the heatmap picked up as highly correlated received a very large coefficient in our models:
- age < 5 : 2nd highest ranked coefficient:	1,781,072.90
- Owner occupied housing units: 4th highest ranked coefficient	959,592.22

Whereas "Total # Homes Owned"	received a coefficient value of 0.00, suggesting that it is not important in determining either of our targets.

As you can see from the plots above, our model does a fairly decent job of predicting the yearly sales given a set of demographic data for a zip code. We were please with this result and continued our efforts.

Rather than utilize a train-test-split or other method to extrapolate our predictions, we felt that the best idea would be to consider the zipcodes that had at least 1 store in them (n=362) vs. the zipcodes that had 0 stores in them (n=459). In doing so, we are now running our model against zipcodes that can be characterized by the demographic data but do not yet contain a liquor store.

Once we ran our models and found their predictions we again sorted our results by Total Yearly Sales with the expectation that based on the demographics of the zip code, these were the strongest contenders for opening future stores.

![](/images/project-03/final_results.png)

Several of our predicted zip codes duplicated zip codes which we found earlier in this write up, however a few new zip codes have emerged: 52001, 52302, 50265, 50322, 50315, 50021, 52402

According to the mode's predictions, based on the zip code's demographic characteristics, these are all highly attractive markets for expansion. However, please be aware that 52403 already has a a relatively high saturation of stores/sqkm. Additionally, 52001 and 52302 both have relatively low dollars / litre and this suggest that the zip may be less likely to support a higher end, more expensive store.

On the map below are the top 10 zip codes which represent the places which our model predicts the highest Total Yearly Sales. It is showing the details for the highest rated zip code, 50317

![](/images/project-03/Lasso Zips Top 10 Detail for nmbr1.png)

### FURTHER WORK

The budget for our data acquisition did not allow us to obtain demographic data newer than the 2000 census, we would highly recommend that more current data be sought to perhaps more accurately reflect current demographic characteristics.

There were a few other questions which arose from the project. We struggled with the fact that our model predicted negative total sales and volume and we'd like to further investigate why that is.

In addition, we'd like to better understand the actual price to the consumer at each store to better understand where the most PROFITABLE stores are located, not simply the stores that are spending the most on purchasing inventory (which is essentially what we've done).

We'd like to better understand if our assumptions made about inventory to sale is reasonable. Perhaps look at whether we can determine areas for socially responsible outreach regarding drinking responsibly.

Can we geocode the stores to a more granular scale and consider the actual commercial areas they reside in? Can we interpolate an optimum mix of products? Can we make decisions about ramping up stock based on seasons?

### POSTSCRIPT

There are many ways to approach the problem of recommending locations for expansion. Certainly there are professional consultant teams with the sole responsibility of doing this type of analysis for corporate expansion. We chose an approach that made sense to us given the data available. We feel confident that there are some valid results which can be drawn from our decisions. But we are also aware that there may be certain aspects and variables which we didn't consider or things that we did include that we shouldn't have.

These are the some of the big questions which face the Data Scientist. To be able to see the various angles, understand the relationships, consider the implications of their decisions. They need to be able to support and defend their approach. They need to create a method which can be tweaked, tuned and reproduced.

These are the struggles. These are a few of the driving forces behind my interest in becoming a productive member of the Data Science community.
