
body ='''Dear All,


We will welcome **NAME** (**AFFILIATION**) for colloquium this Thursday at 2:30p (**DATE**). You can find the abstract for **PRONOUN** talk, **TITLE**, below.

**FIRSTNAME** will be in the department **TIMES**; please sign up to meet with the speaker and for colloquium dinner via this (Google Sheet)[**LINK**]

Cheers,

Erin & Lea

(on behalf of the colloquium committee)

--- --- --- --- --- --- --- --- ---

**ABSTRACT**

'''

pronoun_sub_d = {'his':'he', 'her':'she', 'their':'they'}
pronoun_obj_d = {'his':'him', 'her':'her', 'their':'them'}

body_request = '''Hello!

Our colloquium speaker, **NAME**, has indicated that **PRONOUN_SUB** would especially like to meet with you during **PRONOUN_POS** visit this week. 
Please consider signing up to meet with **PRONOUN_OBJ** during **PRONOUN_POS** department visit via this (Google Sheet)[**LINK**]

Cheers,

Erin
'''

def write_colloquium_email(name, affiliation, date, times, pronoun, link, title, abstract, firstname=None):
    email = body.replace('**NAME**', name)
    if firstname is None:
        firstname = name.split(' ')[0]
        print(f'[Warning] Using the first name {firstname}!\n')
    email = email.replace("**FIRSTNAME**", firstname)
    email = email.replace("**AFFILIATION**", affiliation)
    email = email.replace("**DATE**", date)
    email = email.replace ('**TIMES**', times)
    if pronoun not in ['her','his','their']:
        print('[Warning] double check that pronoun is possessive!' )
    email = email.replace ('**PRONOUN**', pronoun)
    email = email.replace ('**TITLE**', title)
    email = email.replace('**LINK**', link)
    email = email.replace('**ABSTRACT**', abstract )
    return email

def write_colloquium_subject ( name, date ):
    subject = f'Colloquium Sign-up: {name} ({date})'
    return subject

def write_meeting_request ( name, pronoun, link ):
    pronoun_pos = pronoun
    pronoun_sub = pronoun_sub_d[pronoun]
    pronoun_obj = pronoun_obj_d[pronoun]
    
    email = body_request.replace('**NAME**', name)
    email = email.replace('**PRONOUN_SUB**', pronoun_sub )
    email = email.replace('**PRONOUN_OBJ**', pronoun_obj )    
    email = email.replace('**PRONOUN_POS**', pronoun_pos )    
    email = email.replace('**LINK**', link )    
    return email