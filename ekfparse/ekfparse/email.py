import os

class BatchEmail (object):
    def __init__ (self, template, title, savedir=None):#'/scratch/emails/'):
        if savedir is None:
            savedir = os.environ['HOME'] + '/scratch/emails/'
        self.template = open(template,'r').read()
        self.title = title
        self.al_title = ''.join(ch for ch in title if ch.isalnum())
        self.savedir = savedir
        if not os.path.exists(savedir):
            os.makedirs ( savedir )
        
    def address ( self, firstname, lastname, email ):
        body = self.template.replace("FIRSTNAME", firstname)
        body = body.replace("LASTNAME", lastname)
        body = body.replace("RECIPIENTADDRESS", email)
        body = body.replace("EMAILTITLE", self.title)
        return body

    def seal ( self, name, addressed_email ):
        with open (f'{self.savedir}/to_{name}.emltpl', 'w') as f:
            print(addressed_email, file=f)

    