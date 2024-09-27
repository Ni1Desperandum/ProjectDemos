import sys
from collections import defaultdict
from math import fsum
import math

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)
        self.start_rules_list = list
        self.non_start_rules_list = list

 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)
    
    #This function determines whether the sum of the probabilities for each lhs symbol are close to 1
    def is_probabilisitic(self,lhs_symbol):
        lhs_output_list = self.lhs_to_rules[lhs_symbol]
        sum=0
        for i in range(len(lhs_output_list)):
            sum=sum+(lhs_output_list[i][2])
            isprob=math.isclose(sum,1)   
        return isprob

    #This function sorts rules into those that have that start symbol vs. those that don't
    def sort_rules_list(self):
        non_terminals_list=[]
        start_rules_list=[]
        non_start_rules_list=[]
        for key in self.lhs_to_rules.keys():
            non_terminals_list.append(key)
        for lhs_symbol in non_terminals_list:
            if(lhs_symbol == self.startsymbol):
                for item in self.lhs_to_rules[lhs_symbol]:
                    start_rules_list.append(item)
            else:
                for item in self.lhs_to_rules[lhs_symbol]:
                    non_start_rules_list.append(item)
        return start_rules_list, non_start_rules_list
   

    def num_invalid_non_start_rhs(self, non_term_list):
        """
        This function determines the following:
        1. If the length(rhs) == 1, verify that it's a single terminal 
        2. If the length(rhs) == 2, verifty that it's two non-terminals
        If neither of these conditions is met, return false
        """
        num_invalid_rhs_items=0
        start_rules_list = self.sort_rules_list().index(0)
        non_start_rules_list = self.sort_rules_list().index(1)
        for item in non_start_rules_list:
            if len(non_start_rules_list[item][1]) is not 1 and len(non_start_rules_list[item][1]) is not 2: 
                num_invalid_rhs_items=num_invalid_rhs_items+1
            elif len(non_start_rules_list[item][1])== 1 and (non_start_rules_list[item][1][0] in non_term_list): 
                num_invalid_rhs_items=num_invalid_rhs_items+1
            elif (len(non_start_rules_list[item][1])== 2) and (non_start_rules_list[item][1][0] not in non_term_list or non_start_rules_list[item][1][1] not in non_term_list):
                num_invalid_rhs_items=num_invalid_rhs_items+1
            else:
                return num_invalid_rhs_items
        return num_invalid_rhs_items  
        
    def num_invalid_start_rhs(self):
       num_invalid_rhs_items=0
       start_rules_list = self.sort_rules_list().index(0)
       non_start_rules_list = self.sort_rules_list().index(1)
       for item in start_rules_list: 
        #For every production/rule that begins with a start symbol check to see that 1)the is one rhs element and 2) that it's an empty string 
            if len(start_rules_list[item][1]) == 1:
                if(start_rules_list[item][1][0] != ""):
                    num_invalid_rhs_items=num_invalid_rhs_items+1
                else:
                    return num_invalid_rhs_items 
            else: 
                num_invalid_rhs_items=num_invalid_rhs_items+1 #increment the invalid item count because a start symbol with a non-epsilon rhs is not in CNF form
       return num_invalid_rhs_items

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        #Get list of all non_terminals:
        non_terminals_list=[]
        total_invalid_rhs_items=0
        for key in self.lhs_to_rules.keys():
            non_terminals_list.append(key)
        
        #iterate through the rules and see if each rule is probabilistic
        isprob_result_list=[]
        for item in non_terminals_list:
            isprob_result_list.append(self.is_probabilisitic(item)) #Remember that is_probabilistic will take the non-terminal symbol alone 
        if(False in isprob_result_list):
            return False
        else:
            total_invalid_rhs_items=self.num_invalid_non_start_rhs(non_term_list=non_terminals_list)+self.num_invalid_start_rhs(non_term_list=non_terminals_list)
            if total_invalid_rhs_items != 0:
                return False
            else:
                return True
          

if __name__ == "__main__":
    with open('atis3.txt','r') as grammar_file:
        grammar = Pcfg(grammar_file)
        grammar.verify_grammar()
