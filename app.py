from numpy.core.fromnumeric import size
from numpy.lib.type_check import imag
import streamlit as st
import pandas as pd
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
import requests
import altair as alt
from PIL import Image



df = pd.read_parquet('user_data_clean.parquet')


primaryColor = "purple"

### Sidebar
selection = st.sidebar.selectbox("Choose", ["Tweet detection", "Analysis"])
st.sidebar.markdown('##')
st.sidebar.markdown('##')
image_sidebar= Image.open('streamlit_data/lewagon.png')
st.sidebar.image(image_sidebar)
st.sidebar.markdown('##')
st.sidebar.markdown(
    """<hr style="height:2.5px;border:none;color:#333;background-color:#333;" /> """,
    unsafe_allow_html=True)
st.sidebar.markdown('**Le Wagon Data Science Batch 713**')
st.sidebar.markdown('**Alves, Marinescu, Neurauter**')



### BOT PART
if selection == "Tweet detection":
    ### Main part
    ### Top of page - Title, etc
    st.title('The Twitter Bot Recognizer')
    image_header = Image.open('streamlit_data/twitter-bot-verified.png')
    st.image(image_header, width=707)
    st.markdown("<style>.big-font {font-size:10px !important;}</style>",
                unsafe_allow_html=True)
    st.markdown('<p class="big-font">Image_Source: 2021, The Daily Dot, LLC</p>',
                unsafe_allow_html=True)



    st.header('Is the user a human or a bot?')
    st.subheader("Let's find out!")

    st.markdown(
        """<hr style="height:4.5px;border:none;color:#333;background-color:#333;" /> """,
        unsafe_allow_html=True)

    username = ''

    st.subheader("Insert a Username")
    title = st.text_input('Username', value = username)

    ### Predicted Outcome
    st.write(title, 'is definitely ' )

    #params = dict(key='anything',
    #             pickup_datetime=pickup_datetime,
    #            pickup_longitude=pickup_longitude,
    #           pickup_latitude=pickup_latitude,
    #          dropoff_longitude=dropoff_longitude,
    #         dropoff_latitude=dropoff_latitude,
    #        passenger_count=int(passenger_count))



    #url = 'https://taxifare.lewagon.ai/predict'
    # retrieve the response
    #response = requests.get(url, params=params)

    #if response.status_code == 200:
     #   print("API call success")
    #else:
     #   print("API call error")

    #response.status_code
    #predicted_user = response.json().get("prediction", "no prediction")
    #st.markdown(f'# {predicted_user}')







### ANALYSIS PART

if selection == "Analysis":
    ### Space between headers and graph
    st.header('Further Analysis')

    st.write(
        """Since there are a lot of people on Twitter, posting a lot things it is hard
            to distinguih which Twitter post was written and published by a human or a bot.
            """)

    st.write(
        """Therefore, we developed a machine and deep learning algorithm that detects whether a
            Twitter post was published by a human or a bot. Before you can try our excellent app yourself,
            lets look at some data that shows the differences between humans and bots regarding certain user specific features.
            """)
    st.markdown('##')

    ### Showing scatter plot
    st.subheader('Number of followers in relation to the number of tweets')
    c = alt.Chart(df).mark_circle(size=30).encode(
        x='user_followers_cnt',
        y='user_tweet_count',
        color='target',
        tooltip=['user_followers_cnt', 'user_tweet_count', 'target'])

    st.altair_chart(c, use_container_width=True)

    st.markdown(
        """The displayed graph shows a very clear relationship between the number of followers
                for a human and a bot in relation to the number of posted tweets.
                A Twitter account run by a human has considerably more followers than a Twitter account run
                by a bot. On the other hand, it is very clear that bot accounts publish more tweets while at
                the same time have a smaller share of followers compared to humans.
                """)

    ### Bar charts
    st.markdown('##')
    st.subheader(
        'Different user specific features for different kind of users')
    image = Image.open('streamlit_data/user_features.png')
    st.image(image, caption='User Features')

    st.markdown(
        """With the displayed bar charts, we can see that different profile types (humans & bots)
                have different outcomes in regards of specific user features.
                The first graph clearly represents, humans generally have more followers compared to bots.
                On the other hand, bots and humans almost follow the same amount of people.
                Also, not very surprising humans and bots like to post a lot of stuff.
                If we want to find a variable that clearly distinguishes between bot and human, we can look
                at the number of lists a user joined. This number represents for example the number of
                'groups' a user is part of.""")
    st.markdown('##')

    ### Wordcloud
    st.subheader('Who uses which words?')



    st.subheader("Wordcloud Bots")
    image_cloud_bot = Image.open('streamlit_data/wordcloud_bot_500.png')
    st.image(image_cloud_bot,
             caption='Most used words in tweets by bots',
             width=666)

    st.subheader("Wordcloud Humans")
    image_cloud_human = Image.open(
            'streamlit_data/wordcloud_human_500.png')
    st.image(image_cloud_human,
             caption='Most used words in tweets by humans',
             width=666)
