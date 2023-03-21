import ROOT as rt
import pandas as pd

from python.workflow import parallelize
from python.io import mkdir
from stage2.fit_plots import plot
from stage2.fit_models import chebyshev, doubleCB, bwZ, bwGamma, bwZredux, bernstein

rt.RooMsgService.instance().setGlobalKillBelow(rt.RooFit.ERROR)


def run_bias(client, parameters):
     datacard_name = "datacard_cati_ggh.txt"
     npdfs = 7
     ncats = 5
     args = []
     #for ipdf in range(npdfs):                                                                                                                      
     #arg = "combine "+ datacard_name.replace("i",str(ipdf))+" -M GenerateOnly -L HMuMuRooPdfs_cc.so --setParameters pdf_index_ggh="+str(ipdf)+" --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams --toysFrequentist -t 1000 --expectSignal 1 --saveToys -m 125 --freezeParameters pdf_index_ggh"                                                                                                                
          #print(arg)                                                                                                                                
     #     args.append(arg)                                                                                                                          
     cpuinfo = open("/proc/cpuinfo","r")
     cores = sum([(1 if match("^processor\\b.*",l) else 0) for l in cpuinfo])
     print(cores)
     if(cores<35):
          jobs = cores
     else:
          jobs = 35
     print(f'Will run with {jobs} jobs (one per core)')
     workers = []
     for j in range(jobs):
          ipdf = int(j/5)
          icat = j%5
          #print(ipdf, icat)                                                                                                                         
          name = "bias"
          arg1 = "combine "+ datacard_name.replace("i",str(icat))+" -M GenerateOnly -L HMuMuRooPdfs_cc.so --setParameters pdf_index_ggh="+str(ipdf)+" --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams --toysFrequentist -t 1000 --expectSignal 1 --saveToys -m 125 --freezeParameters pdf_index_ggh -n "+ f'_{name}.cat{icat}.pdf{ipdf} > /depot/cms/users/purohita/Hmm_Run3/Bias_logs_26Apr/GenerateOnly_{name}.cat{icat}.pdf{ipdf}.log; mv higgsCombine_{name}.cat{icat}.pdf{ipdf}.GenerateOnly.mH125.123456.root /depot/cms/users/purohita/Hmm_Run3/Bias_outputs_26Apr/;'
          print(arg1)
          args = []
          toyfile_name = f'/depot/cms/users/purohita/Hmm_Run3/Bias_outputs_26Apr/higgsCombine_{name}.cat{icat}.pdf{ipdf}.GenerateOnly.mH125.123456.root'
          for pdf in range(npdfs):
               arg2 = "combine "+ datacard_name.replace("i",str(icat))+" -M FitDiagnostics -L HMuMuRooPdfs_cc.so --setParameters pdf_index_ggh="+str(pdf)+" --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams --toysFile "+toyfile_name+" -t 1000 --rMin -1 --rMax 4 --freezeParameters pdf_index_ggh -m 125 --skipBOnlyFit -n "+f'_{name}.cat{icat}.orgpdf{ipdf}_fitpdf{pdf} > /depot/cms/users/purohita/Hmm_Run3/Bias_logs_26Apr/FitDiagnostics_{name}.cat{icat}.orgpdf{ipdf}_fitpdf{pdf}.log; mv fitDiagnostics_{name}.cat{icat}.orgpdf{ipdf}_fitpdf{pdf}.root /depot/cms/users/purohita/Hmm_Run3/Bias_outputs_26Apr/; mv higgsCombine_{name}.cat{icat}.orgpdf{ipdf}_fitpdf{pdf}.FitDiagnostics.mH125.123456.root /depot/cms/users/purohita/Hmm_Run3/Bias_outputs_26Apr/;'
               args.append(arg2)
               #print(arg2)                                                                                                                          
          #myargs = ["combine"] + [ "-n", f'{name}.{j}']                                                                                             
          myargs = [a for a in arg1.split()]+[b for a in args for b in a.split()]
          #print(myargs)                                                                                                                             
          print(f'spawning {" ".join(myargs)}')
          workers.append( subprocess.Popen(" ".join(myargs), shell=True) )
     for w in workers:
          w.wait()



