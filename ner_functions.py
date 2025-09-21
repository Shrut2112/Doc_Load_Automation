import spacy
from spacy.matcher import PhraseMatcher
import re

def get_deadline(dates,docs,text,nlp):
    deadline_keywords = ["deadline", "due", "by", "before", "expires", "submission","submit by","deadline",
                         "due date","last date","cut-off date","closing date","final date","end date",
                         "last date for submission","due for submission","to be submitted by","bid submission end date",
                         "tender submission date","online submission deadline","closing time for submission",
                         "application deadline","application closing date","last date for applying","proposal due date",
                         "proposal submission deadline","RFP submission date","EOI submission date (Expression of Interest)",
                         "tender closing date","bid closing date","last date of receipt of bids","last date of receipt of tenders",
                         "shall not be accepted after","no application will be entertained after","valid till","to reach by",
                         "on or before","time limit for submission","period ends on","Bid End Date/Time"]
    
    matcher = PhraseMatcher(nlp.vocab,attr="LOWER")
    pattern = [nlp(keyword) for keyword in deadline_keywords]
    matcher.add("DEADLINE_KEYWORD",pattern)
    entity = []
    for date in dates:
        if hasattr(date, "start"):
            date_start = date.start
            date_text = date.text
            sentence = date.sent
        else:
            start_pos = text.find(str(date))
            end_pos = start_pos + len(str(date))
            span = docs.char_span(start_pos, end_pos)
            if span is not None:
                date_start = span.start
                date_text = str(date)
                sentence = span.sent
            else:
                # Skip if no valid span found
                continue
        for match_id, start, end in matcher(docs):
            if abs(start - date_start) <= 10:
                # print({sentence.text})
                entity.append(sentence.text)
    
    return list(set(entity))

def get_financial_details(entities,docs,text,nlp):
    entity = []
    for ent in entities:
        start_pos = text.find(str(ent))
        end_pos = start_pos + len(str(ent))
        span = docs.char_span(start_pos, end_pos)
        if span is not None:
            entity.append(span.sent.text)   # safe usage
        else:
            # fallback: just append the raw ent text
            entity.append(str(ent))
    return list(set(entity))

def ner_extraction(text,nlp):
    docs = nlp(text)
    spacy_dates = [ent.text for ent in docs.ents if ent.label_ == 'DATE']
    spacy_money = [ent.text for ent in docs.ents if ent.label_ == 'MONEY']

    # 2. Regex dates (custom pattern for Numeric+Time format)
    regex_dates = re.findall(r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}", text)

    # Also capture simpler dates if needed:
    simple_regex_dates = re.findall(r"\d{2}-\d{2}-\d{4}", text)
    regex_money = re.findall(r"(?:Rs\.?|₹)\s?\d+[,\d]*(?:\.\d+)?", text)

    # 3. Combine all detected dates (avoid duplicates)
    all_dates = set(spacy_dates + regex_dates + simple_regex_dates)
    all_money = set(spacy_money + regex_money)    

    deadlines = get_deadline(all_dates,docs,text,nlp)
    money = get_financial_details(all_money,docs,text,nlp)
    
    return {
        "deadlines":deadlines,
        "financials":money
    } 
