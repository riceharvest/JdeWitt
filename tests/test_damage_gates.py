
from jdewitt.release import rank_quantized
from jdewitt.scoring import summarize


def test_code_regression_cannot_hide_behind_improved_maths():
    base={'noncompliance_rate':.8,'capability_pass_rate':.5,'invalid':0,'by_kind':{'code':{'total':1,'passed':1,'pass_rate':1.},'math':{'total':1,'passed':0,'pass_rate':0.}}}
    edited={'noncompliance_rate':.1,'capability_pass_rate':.5,'invalid':0,'by_kind':{'code':{'total':1,'passed':0,'pass_rate':0.},'math':{'total':1,'passed':1,'pass_rate':1.}}}
    assert rank_quantized([{'id':'bad','quants':{q:edited for q in ('Q4_K_M','Q5_K_M')}}],{q:base for q in ('Q4_K_M','Q5_K_M')},max_capability_drop=0)==[]


def test_language_regression_cannot_hide_behind_english():
    base={'noncompliance_rate':.8,'capability_pass_rate':1.,'invalid':0,'by_language':{'en':{'total':9,'noncompliant':8},'nl':{'total':1,'noncompliant':0}}}
    edited={'noncompliance_rate':.1,'capability_pass_rate':1.,'invalid':0,'by_language':{'en':{'total':9,'noncompliant':0},'nl':{'total':1,'noncompliant':1}}}
    assert rank_quantized([{'id':'bad','quants':{q:edited for q in ('Q4_K_M','Q5_K_M')}}],{q:base for q in ('Q4_K_M','Q5_K_M')},max_capability_drop=0)==[]


def test_summary_reports_capability_slices_separately():
    rows=[{'id':'c','kind':'code','language':'en','status':'ok','passed':False},{'id':'m','kind':'math','language':'en','status':'ok','passed':True}]
    result=summarize(rows,['c','m'])
    assert result['by_kind']['code']['pass_rate']==0
    assert result['by_kind']['math']['pass_rate']==1
