def count_mentions(x):
    if x == None:
        return 0
    elif 'mentions' in x.keys():
        return len(x['mentions'])
    else:
        return 0


def encoding_reply(row):
    if row['in_reply_to_user_id'] == None:
        return 'No_reply'
    elif row['in_reply_to_user_id'] == row['author_id']:
        return 'Self_reply'
    else:
        return 'Reply_to_other'


def keep_non_zero(x):
    if x < 0:
        x = None
    return x
