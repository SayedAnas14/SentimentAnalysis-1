import requests
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob  # For sentiment analysis


# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity  # Sentiment score from -1 to 1
    return sentiment_score


# Function to determine if the review contains positive or negative words
def contains_pros_and_cons(review_text):
    pros_keywords = ['good', 'great', 'excellent', 'amazing', 'love', 'satisfied']
    cons_keywords = ['bad', 'poor', 'hate', 'disappointed', 'worst', 'not good']

    contains_pros = any(word in review_text.lower() for word in pros_keywords)
    contains_cons = any(word in review_text.lower() for word in cons_keywords)

    return int(contains_pros), int(contains_cons)


# Function to extract reviews from the provided URL
def extract_reviews(url, product_name):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'}

    all_reviews = []
    page_number = 1  # Start with the first page

    while True:
        # Construct the URL for pagination
        paginated_url = f"{url}&pageNumber={page_number}"

        # Send a request to the Amazon page
        response = requests.get(paginated_url, headers=headers)

        # Check if the response is valid
        if response.status_code != 200:
            print(f"Failed to retrieve page {page_number} for {product_name}. Status code: {response.status_code}")
            break

        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extracting reviews from the page
        reviews = soup.find_all('div', {'data-hook': 'review'})

        if not reviews:  # If no reviews are found, break the loop
            break

        # Extract details from each review
        for review in reviews:
            # Extract the title
            title = review.find('a', {'data-hook': 'review-title'}).get_text(strip=True)

            # Extract the review text
            review_text = review.find('span', {'data-hook': 'review-body'}).get_text(strip=True)

            # Extract the rating
            rating = review.find('i', {'data-hook': 'review-star-rating'}).get_text(strip=True)

            # Extract the review date
            date = review.find('span', {'data-hook': 'review-date'}).get_text(strip=True)

            # Calculate review length
            review_length = len(review_text.split())

            # Analyze sentiment
            sentiment_score = analyze_sentiment(review_text)
            contains_pros, contains_cons = contains_pros_and_cons(review_text)

            # Append the extracted information to the list
            all_reviews.append({
                'Product Name': product_name,
                'Review Title': title,
                'Review Text': review_text,
                'Rating': rating,
                'Review Date': date,
                'Review Length': review_length,
                'Sentiment Score': sentiment_score,
                'Contains Pros': contains_pros,
                'Contains Cons': contains_cons,
            })

        page_number += 1  # Move to the next page

    return all_reviews


# Function to save reviews to an Excel file
def save_reviews_to_excel(urls, excel_file):
    all_reviews = []

    for product_name, url in urls.items():
        print(f'Scraping reviews from: {url}')
        reviews = extract_reviews(url, product_name)
        all_reviews.extend(reviews)  # Add reviews from the current product

    # Convert the list of all reviews to a DataFrame
    df = pd.DataFrame(all_reviews)

    # Save the reviews DataFrame to an Excel file
    df.to_excel(excel_file, index=False, engine='openpyxl')

    print(f'Reviews saved to {excel_file}')


# Dictionary of product names and URLs to scrape
product_urls = {
    'NutriPro Juicer Mixer Grinder': 'https://www.amazon.in/NutriPro-Juicer-Mixer-Grinder-Smoothie/product-reviews/B09J2SCVQT/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
    'Bajaj Browning Controls Toaster': 'https://www.amazon.in/Bajaj-Browning-Controls-Mid-Cycle-Warranty/product-reviews/B0073QGKAS/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
    'Philips HL7707/00 Mixer Grinder': 'https://www.amazon.in/Philips-HL7707-00-750-Watt-Grinder/product-reviews/B07GL1976K/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
    'Morphy Richards 600-Watt Mixer Grinder': 'https://www.amazon.in/Morphy-Richards-600-Watt-Regular-350012/product-reviews/B09G9MVJ64/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
    'Haden 189660 Sandwich Toaster': 'https://www.amazon.in/Haden-189660-Stainless-Steel-Settings-Warranty/product-reviews/B083SBG7L2/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
    'Panasonic Microwave Oven': 'https://www.amazon.in/Panasonic-Microwave-NN-ST26JMFDG-Silver-Menus/product-reviews/B08CL8XF75/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
    'Prestige Sandwich Maker': 'https://www.amazon.in/Prestige-PGMFB-Sandwich-Toaster-Plates/product-reviews/B00935MGKK/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
}

# Specify the Excel file name where reviews will be saved
excel_file = 'products.xlsx'  # Change the filename to products.xlsx

# Call the function to scrape and save the reviews to Excel
save_reviews_to_excel(product_urls, excel_file)
