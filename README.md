# Email_Sentiment_Analysis
This code do sentiment analysis on Enron_email dataset.

It get an email in JASON format, and then get you this result as output:

▪ the sentiment of the Email body.
▪ A boolean flag indicating if the Email body is related to Enron's oil & gas
business.
▪ If the email is related to Enron's oil & gas business, then the output also lists
each Person or Organization mentioned

In the Following, I described the instruction of using this repository. 
# Download the dataset

To download the dataset , run these following line in your terminal:

wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
tar -xvzf enron_mail_20150507.tar.gz -C data/

To find more information on the Enron dataset use this link : https://www.cs.cmu.edu/~enron/

# Install Required Library

To run the code, we need some libraries, by running the setup.sh code , all the libraries have been installed.
The names of all libraries have been written in req_libraries.txt file. 

You need to make the setup.sh script executable before running the ./setup.sh. Run the following command in your terminal:

chmod +x setup.sh

# Create Models

Before running the api, run the main.py. This code make some model that are used in the final api. Also, it managed the dataset and you can find the related code there. 
some different methods have been implemented there for sentiment analysis. But at the final step just two of them are selected. The other method such as BERT and transformer have commented to reduce computational resourcses. 

# Run API

To creat an endpoint api , Flask library has used. to Run api follow this instruction: 

pyhton email_analyser_API.py

In a new terminal enter your input as Jason format. Follow the provided example to POST your request to API. 




