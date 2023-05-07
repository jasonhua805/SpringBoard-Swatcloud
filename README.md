# Introduction:
SwatCloud (d.b.a SiliconValley4U) is a small privately owned company in San Ramon, CA which provides educational services and resources to high school and college students with a particular focus on preparing its students for a future career in the technology industry.

The company wanted to implement its own recommendation system that would match its students with active job listings that best fit each person’s individual background and experience. Because each student actively posts projects & blog posts to the Swatcloud website, the company wanted to incorporate these accomplishments through an in-house model instead of just relying on a third-party job board that usually focuses on just a resume (e.g. LinkedIn & Indeed)

# Data Wrangling - HTML Scraping:
The first task at hand is to construct a job listing board. Initially, we attempted to scrape the job listings from LinkedIn and Indeed but these websites block attempts to use HTML scraping on their websites and explicitly prohibit such action in their terms of agreement. The workaround solution to this was to go to tech company career websites and scrape jobs directly from the source. 

The scraping process is done by using BeautifulSoup and Selenium Webdriver. The URL for each company’s careers page is fed into the soup and then the job titles and job descriptions are extracted by identifying all relevant text underneath the key tags/classes that these features are stored under. Note that the specific syntax for each company differs depending on the webpage’s HTML layout, but the overall process flow remains the same. However, this means that each time a new company is added to the database it will require manual effort to repurpose the scraping code to specifically handle the respective career page. Additionally, should a company’s webpage change anything about its HTML layout, it would also require manual effort to refresh its respective extraction function before it can work again. See below for an example using Google:

![image](https://user-images.githubusercontent.com/70826496/236658981-156e4b6e-0a39-4c91-b443-76affdecd80c.png)


Once this process is completed for each company, the outputs are merged into a single dataframe and saved as a csv file to use in our following steps (each company’s output is also saved individually as separate csv files for convenience). See below for the result:

![image](https://user-images.githubusercontent.com/70826496/236658985-09e0db03-2870-421c-a0b3-da69d7b74b35.png)


The main shortfall to our workaround approach is that it will leave out several job listings from smaller companies and/or non-tech companies with technology roles, but nevertheless this process provides a solid foundation upon which we build the basic version of our model. Future iterations of this project can easily improve by adding more companies and/or increasing the scope of our available job listings.






# Exploratory Data Analysis & Data Preprocessing - Industry Labeling:

Once we have our job descriptions ready, our next step is to preprocess our job descriptions so that it’s ready to feed into our recommendation models. We also add an industry label to each job listing in this step which will be a key feature in our NLP model.

We remove blank spaces and special characters from the job descriptions and then use WordNetLemmatizer and NLTK’s stopwords package to polish the data so that only the keywords of interest in each description are being used in our model. 

Because there are a variety of different jobs for each company, we use CountVectorizer to identify what the most common words are in our job titles. The results are shown below:

![image](https://user-images.githubusercontent.com/70826496/236658993-238e0b55-c032-49ac-9ee6-4bcfcf49ab1e.png)


This gives us a high level overview of what types of industries our job board is composed of. We can then start creating different industry labels to encompass these job titles and assigning specific keywords to each of these industries. To label all of our jobs with an industry category, there are two steps:
Identify if a job title has any of the specific keywords used in our industry labels. If so, then assign that industry to that job.
If there are no keywords identified in the title, then the code will loop through the job description and assign an industry label to the job based on which industry’s keywords showed up the most in the description.

Our final industry counts and the resulting data frame looks as follows:

![image](https://user-images.githubusercontent.com/70826496/236658994-3ef59966-2cd2-4065-ab57-2df63d082f04.png)![image](https://user-images.githubusercontent.com/70826496/236658997-8683e417-2b63-452b-849c-ab7db294722c.png)



# Recommendation System Part 1 - Direct Matching to Job Listings

Our recommendation system is actually comprised of two different types of models. The first model matches users to their most suitable job listings based on the cosine similarity between their input (e.g. the user’s resume, blog post content & tags, and overall profile on SwatCloud’s website) and the job database. 

See below for an example of the output - this is a test user who is a current software engineer at Amazon but has an extensive background working in and studying data science

![image](https://user-images.githubusercontent.com/70826496/236658999-01aeb9a9-2c72-4a5d-ae9e-7f0dc2011c01.png)








# Recommendation System Part 2 - NLP Model

The second model in our system is an NLP model which predicts which industries are the most suitable for SwatCloud’s users. We implement a basic Keras sequential model with 4 layers as follows:

![image](https://user-images.githubusercontent.com/70826496/236659003-305f3c92-7d6b-4e2a-9b99-c5a7ff4e2880.png)


To prepare our input for this model, we take our job dataset and tokenize the job descriptions (using a vocabulary size of 10,000) and then split it into training, validation, and test sets. 

Before we begin training the model, we perform cross validation to find the optimal hyperparameters for the following:
Embedding Layer:
Input dimension ranging from 10,000 to 100,000 vocabulary size
Output dimension ranging from 32 to 512 vector length
Convolutional 1D Layer:
Number of output filters ranging from 32 to 512
Kernel size ranging from 1 to 10
We use the ReLu activation function here which is standard for intermediate layers in neural network models.
Dense Layer(s):
Number of dense layers to add ranging from 1 to 3
Number of output units in each layer ranging from 32 to 512
Activation function (one of ReLU, Tanh, or Softmax)
Lastly, we tune whether or not to add a dropout layer and then the learning rate of the model as final hyperparameters. The model is compiled with the Adam optimizer and our model then performs the cross validation.


We run 20 epochs through the training & validation set and then evaluate on our test set. The results are shown below:

![image](https://user-images.githubusercontent.com/70826496/236659009-a7e89197-e7bc-4f0e-9ba1-5f66705c9de6.png)
![image](https://user-images.githubusercontent.com/70826496/236659013-731af0f9-5173-4efa-8409-26e9e534bcfc.png)


Although the training accuracy was able to reach ~88%, the accuracy on the test set was only ~65% suggesting that the model is likely overfitting and further improvements should focus on the data quality rather than the model itself. One area of improvement that immediately stands out is the method in how we delineate and select our industry labels. These categories were manually defined based on human assessment of the most common words in the job titles (as demonstrated in the Exploratory Data Analysis & Data Preprocessing section) combined with intuition on how these words are related to the most common industries in today’s job marketplace.

What exact industries we include and/or how we define them are both significant features we can revisit, but even something more derivative like the number of industries we select also matters. For example, we rerun this model on the same dataset but instead of using the seven industry labels as before, this time we only define each job as being a “Technology” job or a “Non-Technology” job.

![image](https://user-images.githubusercontent.com/70826496/236659020-f8f55578-d8aa-470d-bf53-ff6aee195af4.png)
![image](https://user-images.githubusercontent.com/70826496/236659021-58641850-e2a3-4ad1-a3d8-7689f29f0e3c.png)


Immediately we can see that the model has a higher accuracy on the test set compared to our original dataset. Of course, the practicality of this second dataset is not nearly at the same level (i.e. classifying a user into one of several specific industries like “Software Engineering” or “Marketing” is far more useful than simply recommending whether a user fits a “tech” job or a “non-tech” job), but this example demonstrates the potential that future iterations of this project can improve upon.

We now test this model with our sample users. Here is the same software engineer from Amazon with the data science background which we used earlier in our cosine similarity model.

![image](https://user-images.githubusercontent.com/70826496/236659028-0a11fd31-ba3e-4c6f-aef8-a67bae86278c.png)

The model correctly identifies that his strongest fit is in the Data Analysis sector and also suggests that he is a surprisingly good match for marketing as well. Intuitively we could reason that this is sensible for most professionals working in data who would have sharp business acumen to solve similar problems that marketing professionals would also address.

Here is another test user whose background is a Senior Marketing Analytics Manager at a tech start up company:

![image](https://user-images.githubusercontent.com/70826496/236659032-e48b9efb-2043-4066-b7ca-64a0d14db91e.png)


And below is a fringe case where we input an investment banker’s resume - this individual has no background whatsoever in tech and his experience is very dissimilar to any jobs from the companies that our job database covers (even the non-technology roles in these companies [e.g. Marketing, Sales, etc.] would be very different than this banker’s background):

![image](https://user-images.githubusercontent.com/70826496/236659036-fff0440d-b7bc-478f-a164-3ab63fa48def.png)


The model does recognize that Operations & Finance is a decent fit, but ultimately recommends Software Engineering as the “best match”. This is likely because Software Engineering roles are the most abundant in our database so in an extreme case that the model hasn’t learned very well, it would be biased towards classifying into Software Engineering. This is another area of improvement - the need to gather a wider breadth of data so that the model gains a deeper understanding of the different types of professionals in these industries.

# Final Thoughts & Wrap Up:
After completing our modeling stage and discussing the results with SwatCloud management, the notebooks were deployed onto the company’s website through their Flask production environment.

The company plans to run the HTML scraping script each weekend to refresh the job listings database on a weekly basis and then implement the recommendation models to automatically be an available feature on each of its users’ profile pages. The NLP model will actually target SwatCloud’s younger users (1st & 2nd year in college/university) because its output for an “industry match” is more useful to clients that are still in the exploration stage of their career. On the other hand, the cosine similarity model will focus on SwatCloud’s older users (3rd & 4th year & post-graduate) who are more interested in this model’s direct recommendations for active job listings they can immediately prepare and apply for.

