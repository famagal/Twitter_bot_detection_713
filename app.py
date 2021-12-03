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



    st.subheader("Insert a Username")

    name = st.text_input('Username')

    if name:

        url = f'https://api-twitter-bot-detection-project-enpnocbhoq-uc.a.run.app/predict?username={name}'


        # retrieve the response
        response = requests.get(url)

        predicted_user = response.json()

        if 'oops' in predicted_user.keys():
            st.markdown('Oops we could not find this user. Please try again with different username.')


        elif predicted_user['tweet_level_prediction'] == 'could not fetch tweets for the specified user':
            st.markdown('''We could not fetch any tweets for this user.
                        Predictions will be made solely on user-lever data and therefore, might not be very accurate.''')
            user_pred = predicted_user['user_level_prediction']
            if user_pred == '[0]':
                bot_pred = round((1 - float(predicted_user['bot_proba_user'])) * 100)
                pred = 'human'
            else:
                bot_pred = round(float(predicted_user['bot_proba_user']) *100)
                pred = 'bot'


            st.markdown("<style>.result1-font {font-size:20px !important;}</style>",
                        unsafe_allow_html=True)
            st.markdown(f'''<p class="result1-font">{name} is propably a {pred}.</p>''',
                        unsafe_allow_html=True)

            st.markdown(f'''
                        Accordingly to user-level data we can say with ***{bot_pred}%*** certainty that
                        the chosen user is a ***{pred}***
                        ''')

        else:
            if predicted_user['user_level_prediction'] != predicted_user['tweet_level_prediction'][0]:
                if predicted_user['user_level_prediction'] == '0':
                    user_pred = 'human'
                    user_proba = bot_pred = round((1 - float(predicted_user['bot_proba_user'])) * 100)
                else:
                    user_pred = 'bot'
                    user_proba = round(float(predicted_user['bot_proba_user']) * 100)

                st.markdown(f'''Oops, our algorithm is confused. However, most likely we can say with ***{user_proba}%*** certainty that
                        the chosen user is a ***{user_pred}***.''')
            else:

                if predicted_user['user_level_prediction'] == '0':
                    user_pred = 'human'
                    user_proba = bot_pred = round((1 - float(predicted_user['bot_proba_user'])) * 100)
                else:
                    user_pred = 'bot'
                    user_proba = round(float(predicted_user['bot_proba_user']) * 100)

                if  predicted_user['tweet_level_prediction'] == '0.0':
                    tweet_pred = 'human'
                    tweet_proba = round((1 - float(predicted_user['tweet_proba'])) * 100)
                else:
                    tweet_pred = 'bot'
                    tweet_proba = round(float(predicted_user['tweet_proba']) * 100)


                st.markdown("<style>.result2-font {font-size:20px !important;}</style>",
                            unsafe_allow_html=True)
                st.markdown(
                    f'''<p class="result2-font">{name} is definitely a {user_pred}.</p>''',
                    unsafe_allow_html=True)

                st.markdown(f'''
                            Accordingly to user-level data we can say with ***{user_proba}%*** certainty that
                            the chosen user is a ***{user_pred}***.
                            ''')
                st.markdown(f'''
                            Accordingly to tweet-level data we can say with ***{tweet_proba}%*** certainty that
                            the chosen user is a ***{tweet_pred}***.
                            ''')


        st.markdown("<style>.source-font {font-size:12px !important;}</style>", unsafe_allow_html=True)
        st.markdown(
                    '''<p class="source-font"> * If you want to know more about the differences between user-level -and tweet-level data,
                you can go to our analysis part.
                Just click on the drop down menu on the top left corner of the page.</p>''',
                    unsafe_allow_html=True)


### ANALYSIS PART

if selection == "Analysis":
    ### Space between headers and graph
    st.header('Further Analysis')
    st.subheader('Work in progress..')


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
