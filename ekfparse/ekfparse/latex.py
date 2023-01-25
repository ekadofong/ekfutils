import sys
import bibtexparser
from pylatexenc.latex2text import LatexNodes2Text
from .journal_names import journal_names

names = {'fa_preprint':'First-Author Preprints',
         'fa_published':'First-Author Refereed Publications',
         'ca_preprint':'Co-Author Preprints',
         'ca_published':'Co-Author Refereed Publications',
         'other':'Conference Proceedings and Other Works'}

def bib2CV ( bibtex_file, authorname="Kado-Fong", enumerate=True, file=None ):
    if file is not None:
        writefile = open(file, 'w')
    else:
        writefile = sys.stdout
        
    with open(bibtex_file,'r') as bf:
        bibdb = bibtexparser.load(bf)
    
    mini_bibs = {'fa_preprint':[],
                 'fa_published':[],
                 'ca_preprint':[],
                 'ca_published':[],
                 'other':[],}
    
    for entry in bibdb.entries:
        tag = determine_entryclass ( entry, authorname )
        if tag == 'remove':
            continue
        text = format_entry ( entry, tag )
        mini_bibs[tag].append(text)
    
    idx = 1
    for key in mini_bibs.keys():   
        if len(mini_bibs[key]) == 0:
            continue     
        print('\n'+names[key], file=writefile)
        print('='*len(names[key]), file=writefile)
        #print(mini_bibs[key])
        if enumerate:
            for entrytext in mini_bibs[key]:
                print(f'{idx}. {entrytext}', file=writefile)
                idx+=1
        else:
            for entrytext in mini_bibs[key]:
                print(f'* {entrytext}', file=writefile)            
    if file is not None:
        writefile.close()        
        
def determine_entryclass ( entry, authorname ):
    tag = None
    # \\ is it an "other" -- no journal?
    if 'journal' not in entry.keys():
        tag = 'other'
        if 'doi' in entry.keys():
            is_zenodo = 'zenodo' in entry['doi'].lower() 
            if is_zenodo:                
                return 'remove' 
        return tag
    else:
        # \\ check if refereed
        preprint = 'arxiv' in entry['journal'].lower()
        # \\ check if vizier because that's listed as the journal :shrug:
        # \\ also remove slides 
        is_vizier = 'vizier' in entry['journal'].lower ()        
        if is_vizier:            
            return 'remove'
    
    
  
                
    # \\ check author order
    authors = entry['author'].replace('{','').replace('}','')    
    firstauthor = authors.split(',')[0]
    is_firstauthor = firstauthor == authorname
    
    
    if is_firstauthor and preprint:
        tag = 'fa_preprint'
    elif is_firstauthor:
        tag = 'fa_published'
    elif preprint:
        tag = 'ca_preprint'
    else:
        tag = 'ca_published'
        
    return tag
            
            
def format_entry ( entry, tag, etal=True, nauthors=3, ncut=4 ):
    l2t = LatexNodes2Text()
    clean = lambda x: x.replace('{','').replace('}','')
    title = entry['title']
    title = l2t.latex_to_text ( title )
    
    
    authors = l2t.latex_to_text(entry['author'].replace (' and', ';'))
    if etal and (len(authors.split(';'))>ncut):
        shortauthors = ';'.join(authors.split(';')[:nauthors])
        authors = shortauthors + ' et al.'
        if tag[:2] != 'fa':
            authors = authors + ', incl. EKF'                    
    
    if 'doi' in entry.keys():
        doi = ", " + entry['doi']
    else:
        doi = ''
    if 'journal' in entry.keys():        
        journal = entry['journal'].replace('\\','')
        if journal in journal_names.keys():
            fmted_journal = journal_names[journal] +', '
        else:
            fmted_journal = journal+', '
    else:
        fmted_journal = ''
        
    if tag == 'other':
        if 'booktitle' in entry.keys():
            fmted_journal = entry['booktitle'] + ", "
    year = entry['year']
    month = entry['month']
    
    text = f'''{title}
      {authors}
      [{fmted_journal}{month} {year}{doi}]'''
    return text
        
        
        
            
    
