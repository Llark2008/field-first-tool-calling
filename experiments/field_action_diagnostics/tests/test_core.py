import copy
import unittest

from experiments.field_action_diagnostics import core

TOOLS=[{'name':'fetch','parameters':{'type':'object','properties':{
    'run':{'type':'string'},'metric':{'type':'string','enum':['return']},
    'limit':{'type':'integer','minimum':1,'default':3}},
    'required':['run','metric'],'additionalProperties':False}}]

class ConversionTests(unittest.TestCase):
    def test_known_optional_preserved_and_optional_absence_does_not_ask(self):
        for fields,want in [({'run':'r1','metric':'return','limit':1},{'run':'r1','metric':'return','limit':1}),
                            ({'run':'r1','metric':'return'},{'run':'r1','metric':'return'})]:
            self.assertEqual(core.convert(TOOLS,{'tool':'fetch','fields':fields}),{'action':'call','name':'fetch','arguments':want})

    def test_all_null_required_fields_asked_without_optional(self):
        self.assertEqual(core.convert(TOOLS,{'tool':'fetch','fields':{'run':None,'metric':None}}),
                         {'action':'clarify','missing_fields':['metric','run']})

    def test_invalid_values_and_incomplete_fields_reject_instead_of_guessing(self):
        for fields in [{'run':'r1'},{'run':'r1','metric':'bad'},{'run':'r1','metric':None,'limit':True},
                       {'run':'r1','metric':'return','limit':None},{'run':'r1','metric':'return','extra':2}]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):core.convert(TOOLS,{'tool':'fetch','fields':fields})

    def test_defaults_unsupported_and_unknown_tool(self):
        tools=copy.deepcopy(TOOLS);tools[0]['parameters']['required'].append('limit')
        self.assertEqual(core.convert(tools,{'tool':'fetch','fields':{'run':'r1','metric':'return','limit':None}})['arguments']['limit'],3)
        self.assertEqual(core.convert(TOOLS,{'tool':None,'fields':{}}),{'action':'unsupported'})
        for obj in [{'tool':'unknown','fields':{}},{'tool':None,'fields':{'run':'r1'}}]:
            with self.assertRaises(ValueError):core.convert(TOOLS,obj)

    def test_first_object_independent_of_bad_action_but_rejects_duplicate_and_truncation(self):
        raw='{"tool":"fetch","fields":{"run":"r1","metric":"return"}}\nBROKEN ACTION'
        self.assertEqual(core.first_object(raw,'stop')['tool'],'fetch')
        for text,reason in [(raw,'length'),('{"tool":null,"tool":"fetch","fields":{}}\n{}','stop')]:
            with self.assertRaises(ValueError):core.first_object(text,reason)

    def test_mapping_prompt_excludes_labels_and_original_request(self):
        fields={'tool':'fetch','fields':{'run':'r1','metric':None}}
        text=str(core.mapping_messages(TOOLS,fields))
        self.assertIn('r1',text)
        self.assertIn('null',text)
        self.assertNotIn('missing_fields": ["metric"]',text)

    def test_exact_mapping_checks_all_fields_types_and_tool(self):
        expected={'action':'call','name':'fetch','arguments':{'run':'r1','metric':'return'}}
        self.assertTrue(core.action_matches(TOOLS,expected,expected))
        self.assertTrue(core.action_matches(TOOLS,expected,dict(expected,arguments={'run':'r1','metric':'return','limit':3})))
        self.assertFalse(core.action_matches(TOOLS,expected,dict(expected,arguments={'run':'r1','metric':'return','limit':True})))
        self.assertFalse(core.action_matches(TOOLS,{'action':'clarify','missing_fields':['run','metric']},{'action':'clarify','missing_fields':['run']}))

if __name__=='__main__':unittest.main()
