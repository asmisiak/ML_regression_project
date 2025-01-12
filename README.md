# Regression Problem - AIrBnB Price Prediction in 3 European Cities

Nightly pricing on Airbnb is a legitimate practice that hosts employ to stay competitive in the short-term rental market. By definition, pricing an Airbnb listing involves structuring rental costs in such a way that they might diverge from what is commonly expected, ultimately seeking to balance occupancy and profitability. The rapid expansion of Airbnb in key European tourism hubs—Barcelona, Paris, and Rome—has been driven by a combination of increased travel demand, liberalized local regulations in some areas, and advancements in technology-based marketplaces. These factors underscore the need to better understand the drivers of Airbnb pricing and how best to forecast them.

In this study, we will investigate whether it is possible to forecast, one year ahead, the average nightly price of Airbnb listings in Barcelona, Paris, and Rome using classic (or “shallow”) Machine Learning methods. As prior research in the short-term rental market suggests, this task is far from trivial.  Accurate predictions of nightly rates may be particularly beneficial not only for hosts aiming to set competitive prices, but also for policymakers and tourism boards monitoring housing market dynamics.

There are many proposed methods in the literature for measuring or defining Airbnb prices. One widely used approach is to track the actual nightly rate displayed on the platform at a given time. Although straightforward, this measure can vary significantly across high and low seasons, and might not always capture discounts or fees that apply only sporadically. Nonetheless, we will use the raw listing price as our key target (endogenous) variable in this study, as it represents the direct cost faced by travelers and the primary revenue source for hosts.

Therefore, the following evaluation metrics for the forecasting problem have been selected: Mean Absolute Error (MAE) and Root Mean Square Error (RMSE). This choice is deliberate: MAE provides a more interpretable measure of average error in currency units, while RMSE penalizes larger individual errors more strongly—an important consideration when trying to anticipate large price fluctuations in highly sought-after tourist seasons. Between these two, RMSE will be of primary importance, as extreme mispricing can have significant impacts on hosts’ income and market fairness.

Ultimately, this problem can be viewed as a form of panel data analysis (covering multiple listings over multiple time periods) that incorporates both cross-sectional and temporal aspects of Airbnb pricing. By exploring this domain, we hope to shed light on how short-term rental prices behave in these iconic European cities, and how Machine Learning can be used to forecast them more effectively.

## Dataset Description

This dataset provides a comprehensive look at Airbnb prices in some of the most popular European cities. Each listing is evaluated for various attributes such as room types, cleanliness and satisfaction ratings, bedrooms, distance from the city centre, and more to capture an in-depth understanding of Airbnb prices on both weekdays and weekends. Using spatial econometric methods, we analyse and identify the determinants of Airbnb prices across these cities.

Our dataset includes information such as:
- room_type- The type of room being offered (e.g. private, shared, etc.). (Categorical)
- room_shared- Whether the room is shared or not. (Boolean)
- room_private- Whether the room is private or not. (Boolean)
- person_capacity- The maximum number of people that can stay in the room. (Numeric)
- host_is_superhost- Whether the host is a superhost or not. (Boolean)
- multi- Whether the listing is for multiple rooms or not. (Boolean)
- biz- Whether the listing is for business purposes or not. (Boolean)
- cleanliness_rating- The cleanliness rating of the listing. (Numeric)
- guest_satisfaction_overall- The overall guest satisfaction rating of the listing. (Numeric)
- bedrooms- The number of bedrooms in the listing. (Numeric)
- dist- The distance from the city centre. (Numeric)
- metro_dist	The distance from the nearest metro station.- (Numeric)
- lng- The longitude of the listing. (Numeric)
- lat- The latitude of the listing. (Numeric)
- is_weekend- Whether the offer referred to weekdays or weekends (Boolean)
- city- city of Airbnb offer (Categorical)

Source: https://www.kaggle.com/datasets/thedevastator/airbnb-prices-in-european-cities/data
This project will cover panel data for 3 cities (Barcelona, Paris, Rome).
