'''from datetime import datetime, date

string_input_with_date = "08-2022"

import pandas as pd
past = datetime.strptime(string_input_with_date, "%m-%Y")
present = datetime.now()
delta = past-present
#print(delta.days)
#print(present,past)
#if past < present:
#    print("ok")
#else:
#    print("no")

today = date.today()
now = datetime.now()
date_time = datetime.now().strftime("%m-%Y")
date_time=datetime.strptime(date_time, "%m-%Y")
#if past < date_time:
#    print("ok")
#else:
#    print("no")


from datetime import datetime
from dateutil.relativedelta import relativedelta

# Returns the same day of last month if possible otherwise end of month
# (eg: March 31st->29th Feb an July 31st->June 30th)
last_month = datetime.now() - relativedelta(months=1)

# Create string of month name and year...
text = format(last_month, '%B %Y')
#print(text)

df = pd.read_csv("/home/dharmendra/hrbotv1/hrbot2/database/employees.csv")
print(df._get_value(0, 'applied'))

# updating the column value/data
df.loc[0, 'applied'] = 3
df.loc[0, 'loss_of_pay'] = 3
  
# writing into the file
df.to_csv("/home/dharmendra/hrbotv1/hrbot2/database/user_auth.csv", index=False)
'''
"""
def handle_reimbursement_submit_doc1(gs_context,current_context):
  gs_context["slots"]["user_reimbursement_confirm1"] = current_context['current_intent']['user_text']
  if gs_context["slots"]["user_reimbursement_confirm1"] == "Submit Doc":
    bill_type1 = gs_context["slots"]["bill_type1"]
    response_json_list =  {
  "card_type": "quick_reply",
  "content_type": "text",
  "text": "Please confirm you have applied for reimbursement of \n1.{}".format(bill_type1),
  "options": [
    {
      "title": "Confirm & Submit"
    },
    {
      "title": "Never Mind"
    }
  ]
}
    help.show_response_card(gs_context,current_context,response_json_list)
    gs_context["past_conversation"] = "reimbursement_submit_doc_done"
    current_context['current_intent']['intent'] = ""
  else:
    handle_greet_intent(gs_context, current_context)
    gs_context["past_conversation"] = ""
    current_context['current_intent']['intent'] = ""
  return    
"""


'''def handle_reimbursement_submit_doc2(gs_context,current_context):
  gs_context["slots"]["user_reimbursement_confirm2"] = current_context['current_intent']['user_text']
  if gs_context["slots"]["user_reimbursement_confirm2"] == "Submit Doc":
    bill_type1 = gs_context["slots"]["bill_type1"]
    bill_type2 = gs_context["slots"]["bill_type2"]
    response_json_list =  {
  "card_type": "quick_reply",
  "content_type": "text",
  "text": "Please confirm you have applied for reimbursement of \n1.{} \n2.{} \n3.{}".format(bill_type1,bill_type2),
  "options": [
    {
      "title": "Confirm & Submit"
    },
    {
      "title": "Never Mind"
    }
  ]
}
    help.show_response_card(gs_context,current_context,response_json_list)
    gs_context["past_conversation"] = "reimbursement_submit_doc_done"
    current_context['current_intent']['intent'] = ""
  else:
    handle_greet_intent(gs_context, current_context)
    gs_context["past_conversation"] = ""
    current_context['current_intent']['intent'] = ""
  return'''    

from datetime import datetime
def str_insert(string, char, index):
    return string[:-index] + char + string[-index:]
def str_insert1(string, char):
    return string + char

def date_validation_new(input_date):
  today_date = datetime.now()
  format = "%d-%m-%Y"
  status = 0
  status1 = 0
  ddmmyyyy = input_date.split('-')
  if len(ddmmyyyy) == 3:
    if (len(ddmmyyyy[0]) == 1 or 2) and (len(ddmmyyyy[1]) == 1 or 2): #and (len(ddmmyyyy[2]) == 2 or 4):
        if len(ddmmyyyy[2]) == 2:
            input_date = str_insert(input_date, str(today_date)[0:2], 2)
            try:
                datetime.strptime(input_date, format)
                status1 = 1
            except ValueError:
                status1 = 0
        elif len(ddmmyyyy[2]) == 4:
            try:
                datetime.strptime(input_date, format)
                status1 = 1
            except ValueError:
                status1 = 0
        else:pass
    else:pass
  elif len(ddmmyyyy) == 2:
    if (len(ddmmyyyy[0]) == 1 or 2) and (len(ddmmyyyy[1]) == 1 or 2): #and (len(ddmmyyyy[2]) == 2 or 4):
        #if len(ddmmyyyy[2]) == 2:
            input_date = str_insert1(input_date, "-{}".format(str(today_date)[0:4]))
            try:
                datetime.strptime(input_date, format)
                status1 = 1
            except ValueError:
                status1 = 0
        #elif len(ddmmyyyy[2]) == 4:
        #    try:
        #        datetime.strptime(input_date, format)
        #        status1 = 1
        #    except ValueError:
        #        status1 = 0
        #else:pass
    elif len(ddmmyyyy[1]) == 1 or 2: #and (len(ddmmyyyy[2]) == 2 or 4):
        #if len(ddmmyyyy[2]) == 2:
            input_date  = str_insert(input_date, str(today_date)[0:2], 2)
    else:pass  
  else:pass
  status2 = 0
  if status1 ==1:
    input_date1=datetime.strptime(input_date, "%d-%m-%Y")
    if input_date1.date() > today_date.date():
      status2 = 1
    else: status2 = 0
  else:pass
  if status2 == 1:
    input_year = datetime.strptime(input_date, "%d-%m-%Y").date().year
    current_year = datetime.now().year
    if input_year==current_year:
      status = 1
    else: status = 0
  else:pass
  return input_date, status


def date_validation_new1(from_date,to_date):
  from_date,status_from = date_validation_new(from_date)
  to_date,status_to = date_validation_new(to_date)
  format = "%d-%m-%Y"
  status = 0
  if status_from ==1 and status_to == 1:
    f_date=datetime.strptime(from_date, format)
    t_date=datetime.strptime(to_date, format)
    print(f_date,t_date)
    if t_date.date() >= f_date.date():
      status = 1
    else: status = 0

  return from_date,to_date,status
#print(date_validation_new('01-22'))
#print(date_validation_new1('02-08-20','5-8-22'))


######## Graph #####
nodes = []
graph = []
nodes_count = 0
def add_node(v):
    global nodes_count, nodes, graph
    if v in nodes:
        print("Node already present in graph")
    else:
        nodes_count = nodes_count + 1
        nodes.append(v)
        for n in graph:
            n.append(0)
        temp = []
        for i in range(nodes_count):
            temp.append(0)
        graph.append(temp)
def add_edge(v1,v2):
    
#def print_graph():
#    for i in range(nodes_count):
#        for j in range(nodes_count):
#            print(graph[i][j],end=" ")
#        print("\n")
    

add_node("a")
add_node("b")
add_node("c")
add_node("d")
print(nodes)
print(print_graph())