import ROOT as rt
import pandas as pd

from python.workflow import parallelize
from python.io import mkdir
from stage2.fit_plots import plot
from stage2.fit_models import chebyshev, doubleCB, bwZ, bwGamma, bwZredux, bernstein

rt.RooMsgService.instance().setGlobalKillBelow(rt.RooFit.ERROR)


def run_Category_Optimizer(client, parameters, df):
    signal_ds = parameters.get("signals", [])
    data_ds = parameters.get("data", [])
    all_datasets = df.dataset.unique()
    signals = [ds for ds in all_datasets if ds in signal_ds]
    backgrounds = [ds for ds in all_datasets if ds in data_ds]
    fit_setups = []
    if len(backgrounds) > 0:
        fit_setup = {
            "label": "background",
            "mode": "bkg",
            "df": df[df.dataset.isin(backgrounds)],
            "blinded": True,
        }
        fit_setups.append(fit_setup)
    for ds in signals:
        fit_setup = {"label": ds, "mode": "sig", "df": df[df.dataset == ds]}
        fit_setups.append(fit_setup)

    argset = {
        "fit_setup": fit_setups,
        "channel": parameters["mva_channels"],
        "category": df["category"].dropna().unique(),
    }
    fit_ret = parallelize(fitter, argset, client, parameters)
    df_fits = pd.DataFrame(columns=["label", "channel", "category", "chi2"])
    for fr in fit_ret:
        df_fits = pd.concat([df_fits, pd.DataFrame.from_dict(fr)])
    # choose fit function with lowest chi2/dof
    df_fits.loc[df_fits.chi2 <= 0, "chi2"] = 999.0
    df_fits.to_pickle("all_chi2.pkl")
    idx = df_fits.groupby(["label", "channel", "category"])["chi2"].idxmin()
    df_fits = (
        df_fits.loc[idx]
        .reset_index()
        .set_index(["label", "channel"])
        .sort_index()
        .drop_duplicates()
    )
    print(df_fits)
    df_fits.to_pickle("best_chi2.pkl")
    return fit_ret


def fitter(args, parameters={}):
    fit_setup = args["fit_setup"]
    df = fit_setup["df"]
    label = fit_setup["label"]
    mode = fit_setup["mode"]
    blinded = fit_setup.get("blinded", False)
    save = parameters.get("save_fits", False)
    save_path = parameters.get("save_fits_path", "fits/")
    channel = args["channel"]
    category = args["category"]

    save_path = save_path + f"/fits_{channel}_{category}/"
    mkdir(save_path)
    optimized_mva_cuts = []
    optimized_max_significance = []
    #Signal fitting
    sig_paths = glob.glob("/depot/cms/hmm/purohita/copperhead/2July2022_checkBDTEvaluation/stage2_unbinned/ggh_201*/ggh_amcPS*.parquet")
    df_sig_col = []
    for path in sig_paths:
        df_sig_col.append(pd.read_parquet(path))

    df_sig = pd.concat(df_sig_col)
    print(df_sig.head())
    df_sig = df_sig[(df_sig['dimuon_mass']>109.9) & (df_sig['dimuon_mass']<140.1)]
    data_paths = glob.glob("/depot/cms/hmm/purohita/copperhead/2July2022_checkBDTEvaluation/stage2_unbinned/ggh_201*/data*.parquet")
    df_data_col = []
    for path in data_paths:
        df_data_col.append(pd.read_parquet(path))
        
    df_data = pd.concat(df_data_col)
    print(df_data.head())
    df_data = df_data[(df_data['dimuon_mass']>109.9) & (df_data['dimuon_mass']<150.1)]

    n_cats = 3
    for icat in range(n_cats):
        output_dict = {
            "sig_eff": [],
            "mva_val": [],
            "significance": [],
        }
        n_points = 50
        for i in range(n_points):
            print("D"*100)
            print("For the "+str(i)+"th point signal eff"+str(1-float(i+1+float(icat)/20.5)/(n_points+1)))
            output_dict["sig_eff"].append(1-float(i+1+float(icat)/20.5)/(n_points+1))
            mva_cuts = weighted_quantile(df_sig['score_bdt_test'].values, 1-np.array([float(i+1+float(icat)/20.5)/(n_points+1)]), sample_weight=df_sig['wgt_nominal'].values)
            mva_cuts = np.concatenate((mva_cuts,np.array(optimized_mva_cuts)))
            output_dict["mva_val"].append(mva_cuts[0])
            #mva_cuts = []
            print("with mva value: "+str(mva_cuts[0]))
            #sys.exit()
            #mva_cuts = []
            df_sig_cats = categorize(df_sig, mva_cuts)
            df_data_cats = categorize(df_data, mva_cuts)
            parameters["datacards_list"] = []
            for cat_i in range(len(mva_cuts)+1):
                print(df_sig_cats[cat_i].shape)
                print(df_data_cats[cat_i].shape)
                outpath_sig = "/depot/cms/hmm/purohita/copperhead/2July_new2022_new_checkBDTEvaluation_25July_CatOpt_5cats/sig_cat"+str(cat_i)+".root"
                dataframe_to_root(df_sig_cats[cat_i], outpath_sig)
                outpath_data = "/depot/cms/hmm/purohita/copperhead/2July_new2022_new_checkBDTEvaluation_25July_CatOpt_5cats/data_cat"+str(cat_i)+".root"
                dataframe_to_root(df_data_cats[cat_i], outpath_data)
                
                """
                client = Client(
                processes=True,
                n_workers=parameters["ncpus"],
                threads_per_worker=1,
                memory_limit="4GB",
                )
                """
                parameters['category'] = "cat"+str(cat_i)
                parameters['input_path_sig'] = outpath_sig
                parameters['input_path_data'] = outpath_data
                #parameters['output_path'] = "/depot/cms/users/purohita/Hmm_Run3/Run3HmmAnalysis/21July2022_checkBDTEvaluation_NewMVABins/"
                parameters['output_path'] = "/depot/cms/hmm/purohita/copperhead/2July_new2022_new_checkBDTEvaluation_25July_CatOpt_5cats/"
                parameters["sig_norm"] = df_sig_cats[cat_i]['wgt_nominal'].sum()
                print("The signal normalization is ",parameters["sig_norm"])
                #print(parameters["sig_norm"])
                parameters["bkg_norm"] = df_data_cats[cat_i]['wgt_nominal'].sum()
                print("The background normalization is ",parameters["bkg_norm"])
                args.doSignalFit = True
                args.doBackgroundFitWithMultiPdf = False
                workflow( parameters)
                args.doBackgroundFitWithMultiPdf=True
                args.doSignalFit = False
                workflow( parameters)
                
                tag = "_" + parameters['process'] + "_" + parameters['category']
                #parameters["sig_file"] = parameters['output_path']+"workspace_sigFit" + tag + args.ext+".root"
                parameters["sig_file"] = "workspace_sigFit" + tag + args.ext+".root"
                #parameters["bkg_file"] = parameters['output_path']+"workspace_BackgroundFitWithMultiPdf" + tag + args.ext+".root"
                parameters["bkg_file"] = "workspace_BackgroundFitWithMultiPdf" + tag + args.ext+".root"
                parameters["cat"] = parameters['process'] + "_" + parameters['category']
                parameters["original_datacard"] = "/home/purohita/Hmm/June23_2020/CMSSW_10_2_13/src/hig-19-006/ggH/ucsd/Phase2_DelphesAnalysis/datacard_cat_ggh_template.txt" 
                parameters["datacard_out_path"] = parameters['output_path']
                parameters["datacard_outfile_name"] = "datacard"+tag+"_"+args.ext+".txt"
                createCombineDatacards(parameters)
                parameters["datacards_list"].append(parameters["datacard_out_path"]+parameters["datacard_outfile_name"])
                sig = runSignificance(parameters)
                print(str(sig)+" is the Significance in "+parameters["cat"])
                    
            parameters["combined_datacard_outfile_name"] = "datacard_combined_ggh"+args.ext+".txt"
            combine_datacards(parameters)
            parameters["datacard_outfile_name"] = parameters["combined_datacard_outfile_name"]
            print("Running combine significance for "+parameters["combined_datacard_outfile_name"]+" datacard")
            combined_sig = runSignificance(parameters)
            if icat!=0:
                print(str(combined_sig-optimized_max_significance[-1])+" is the improvement in combined Significance in all categories")
                output_dict["significance"].append(combined_sig-optimized_max_significance[-1])
            else:
                print(str(combined_sig)+" is the improvement in combined Significance in all categories")
                output_dict["significance"].append(combined_sig)
                
        print(output_dict)
        np.save(parameters['output_path']+'signal_efficiency'+str(icat)+'Category_'+args.ext+'.npy', np.array(output_dict["sig_eff"]))
        np.save(parameters['output_path']+'significance'+str(icat)+'Category_'+args.ext+'.npy', np.array(output_dict["significance"]))
        optimized_mva_cuts.append(output_dict["mva_val"][np.argmax(np.array(output_dict["significance"]))])
        optimized_max_significance.append(max(output_dict["significance"]))
    print(optimized_mva_cuts)
    print(optimized_max_significance)
    np.save(parameters['output_path']+'Optimized_mva_cuts'+str(n_cats)+'Categories_'+args.ext+'.npy', np.array(optimized_mva_cuts))
    np.save(parameters['output_path']+'Optimized_max_significance'+str(n_cats)+'Categories_'+args.ext+'.npy', np.array(optimized_max_significance))

    #df = df[(df.channel == args["channel"]) & (df.category == args["category"])]
    norm = df.lumi_wgt.sum()

    the_fitter = Fitter(
        fitranges={"low": 110, "high": 150, "SR_left": 120, "SR_right": 130},
        fitmodels={
            "bwz": bwZ,
            "bwz_redux": bwZredux,
            "bwgamma": bwGamma,
            "bernstein": bernstein,
            "dcb": doubleCB,
            "chebyshev": chebyshev,
        },
        requires_order=["chebyshev", "bernstein"],
        channel=channel,
        filename_ext="",
    )
    if mode == "bkg":
        chi2 = the_fitter.simple_fit(
            dataset=df,
            label=label,
            category=category,
            blinded=blinded,
            model_names=["bwz", "bwz_redux", "bwgamma"],
            fix_parameters=False,
            store_multipdf=True,
            title="Background",
            save=save,
            save_path=save_path,
            norm=norm,
        )
        # generate and fit pseudo-data
        the_fitter.fit_pseudodata(
            label="pseudodata_" + label,
            category=category,
            blinded=blinded,
            model_names=["bwz", "bwz_redux", "bwgamma"],
            fix_parameters=False,
            title="Pseudo-data",
            save=save,
            save_path=save_path,
            norm=norm,
        )

    if mode == "sig":
        chi2 = the_fitter.simple_fit(
            dataset=df,
            label=label,
            category=category,  # temporary
            blinded=False,
            model_names=["dcb"],
            fix_parameters=True,
            store_multipdf=False,
            title="Signal",
            save=save,
            save_path=save_path,
            norm=norm,
        )
    ret = {"label": label, "channel": channel, "category": category, "chi2": chi2}
    return ret
def categorize(df, mva_cuts):
    df_col = []
    for i in range(len(mva_cuts)+1):
        if(len(mva_cuts)==0):
            df_col.append(df)
            continue
        if(i==0):
            df_i = df[df["score_bdt_test"]<mva_cuts[i]]
        elif(i<(len(mva_cuts))):
            df_i = df[(df["score_bdt_test"]<mva_cuts[i])&(df["score_bdt_test"]>mva_cuts[i-1])]
        else:
            df_i = df[df["score_bdt_test"]>mva_cuts[i-1]]
        df_col.append(df_i)
    return df_col

def categorize_inEta(df, mva_cuts):
    df_col = []
    print(len(mva_cuts)-1)
    for i in range(len(mva_cuts)-1):
        if(len(mva_cuts)==0):
            df_col.append(df)
            continue
        #if(i==0):
        #    df_i = df[df[["mu1_eta","mu2_eta"]].abs().max(axis=1)<mva_cuts[i]]
        #elif(i<(len(mva_cuts))):
        df_i = df[(df[["mu1_eta","mu2_eta"]].abs().max(axis=1)<mva_cuts[i+1])&(df[["mu1_eta","mu2_eta"]].abs().max(axis=1)>mva_cuts[i])]
        #else:
        #    df_i = df[df[["mu1_eta","mu2_eta"]].abs().max(axis=1)>mva_cuts[i-1]]
        df_col.append(df_i)
    return df_col



def dataframe_to_root(df, out_path):
    import root_numpy
    import numpy as np
    print(df[["dimuon_mass", "wgt_nominal"]].values)
    #t = np.array([(df["dimuon_mass"].values), (df["wgt_nominal"].values)], dtype=[("mass", np.float32), ("weight", np.float32)])
    t = np.array([tuple(x) for x in df[["dimuon_mass", "wgt_nominal"]].values.tolist()], dtype=[("mass", np.float32), ("weight", np.float32)])
    print(type(t))
    print(t)
    root_numpy.array2root(t, out_path, treename="tree", mode='recreate')
    return

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def createCombineDatacards(parameters):
    print("Creating a datacard:   ")
    from string import Template
    datacard_template =  parameters["original_datacard"]

    substitutions = {
        "input_file_sig": parameters["sig_file"],
        "input_file_bkg": parameters["bkg_file"],
        "cat_ggh": parameters["cat"],
        "bkg_rate": parameters["bkg_norm"],
    }
    
    with open(datacard_template, "r") as f:
        tmp = f.read()
    custom_text = Template(tmp).substitute(**substitutions)
    out_fullpath = parameters["datacard_out_path"] + parameters["datacard_outfile_name"]
    with open(out_fullpath, "w") as f:
        f.write(custom_text)
    print("Saved datacard here: "+out_fullpath)

def runSignificance(parameters):
    COMBINE_OPTIONS = " --X-rtd FITTER_NEWER_GIVE_UP --X-rtd FITTER_NEW_CROSSING_ALGO --cminDefaultMinimizerStrategy 0 --cminRunAllDiscreteCombinations --cminApproxPreFitTolerance=0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.01 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rtd MINIMIZER_MaxCalls=9999999 --X-rtd MINIMIZER_analytic --X-rtd FAST_VERTICAL_MORPH --cminDefaultMinimizerTolerance 0.01 --X-rtd MINIMIZER_freezeDisassociatedParams"
    text2workspace_arguments = " -m 125"
    setParameters = ""
    #freezeParameters =" --freezeParameters pdf_index_ggh"
    freezeParameters =""
    combine_arguments = (
        " -M Significance -m 125 --expectSignal=1 -t -1" + COMBINE_OPTIONS
    )
    # Convert datacards to ROOT files for a given uncertainty scenario
    to_workspace = "text2workspace.py "+parameters["datacard_out_path"]+parameters["datacard_outfile_name"]+text2workspace_arguments
    subprocess.check_output([to_workspace], shell=True)
    command = "combineTool.py -d "+parameters["datacard_out_path"]+parameters["datacard_outfile_name"]+combine_arguments+setParameters+freezeParameters
    command = command.replace(".txt", ".root")
    value = float(
            subprocess.check_output([command+" | grep Significance:"], shell=True)
            .decode("utf-8")
            .replace("Significance: ", "")
        )
    return value
def combine_datacards(parameters):
    datacard_names = parameters["datacards_list"]
    combine_cards = "combineCards.py "+" ".join(datacard_names)+" > "+parameters["datacard_out_path"]+parameters["combined_datacard_outfile_name"]
    subprocess.check_output([combine_cards], shell=True)


class Fitter(object):
    def __init__(self, **kwargs):
        self.fitranges = kwargs.get(
            "fitranges", {"low": 110, "high": 150, "SR_left": 120, "SR_right": 130}
        )
        self.fitmodels = kwargs.get("fitmodels", {})
        self.requires_order = kwargs.get("requires_order", [])
        self.channel = kwargs.get("channel", "ggh_0jets")
        self.filename_ext = kwargs.get("filename_ext", "")

        self.data_registry = {}
        self.model_registry = []

        self.workspace = self.create_workspace()

    def simple_fit(
        self,
        dataset=None,
        label="test",
        category="cat0",
        blinded=False,
        model_names=[],
        orders={},
        fix_parameters=False,
        store_multipdf=False,
        title="",
        save=True,
        save_path="./",
        norm=0,
    ):
        if dataset is None:
            raise Exception("Error: dataset not provided!")
        if len(model_names) == 0:
            raise Exception("Error: empty list of fit models!")

        ds_name = f"ds_{label}"
        self.add_data(dataset, ds_name=ds_name, blinded=blinded)
        ndata = len(dataset["dimuon_mass"].values)

        for model_name in model_names:
            if (model_name in self.requires_order) and (model_name in orders.keys()):
                for order in orders[model_name]:
                    self.add_model(model_name, category=category, order=order)
            else:
                self.add_model(model_name, category=category)

        # self.workspace.Print()
        chi2 = self.fit(
            ds_name,
            ndata,
            model_names,
            orders=orders,
            blinded=blinded,
            fix_parameters=fix_parameters,
            category=category,
            label=label,
            title=title,
            save=save,
            save_path=save_path,
            norm=norm,
        )
        if store_multipdf:
            cat = rt.RooCategory(
                f"pdf_index_{self.channel}_{category}_{label}",
                "index of the active pdf",
            )
            pdflist = rt.RooArgList()
            for model_name in model_names:
                pdflist.add(self.workspace.pdf(model_name))
            multipdf = rt.RooMultiPdf(
                f"multipdf_{self.channel}_{category}_{label}", "multipdf", cat, pdflist
            )
            # self.add_model("multipdf", category=category)
            getattr(self.workspace, "import")(multipdf)
        if save:
            mkdir(save_path)
            self.save_workspace(
                f"{save_path}/workspace_{self.channel}_{category}_{label}{self.filename_ext}"
            )
        return chi2

    def create_workspace(self):
        w = rt.RooWorkspace("w", "w")
        mass = rt.RooRealVar(
            "mass", "mass", self.fitranges["low"], self.fitranges["high"]
        )
        mass.setRange(
            "sideband_left", self.fitranges["low"], self.fitranges["SR_left"] + 0.1
        )
        mass.setRange(
            "sideband_right", self.fitranges["SR_right"] - 0.1, self.fitranges["high"]
        )
        mass.setRange("window", self.fitranges["low"], self.fitranges["high"])
        mass.SetTitle("m_{#mu#mu}")
        mass.setUnit("GeV")
        # w.Import(mass)
        getattr(w, "import")(mass)
        # w.Print()
        return w

    def save_workspace(self, out_name):
        outfile = rt.TFile(f"{out_name}.root", "recreate")
        self.workspace.Write()
        outfile.Close()

    def add_data(self, data, ds_name="ds", blinded=False):
        if ds_name in self.data_registry.keys():
            raise Exception(
                f"Error: Dataset with name {ds_name} already exists in workspace!"
            )

        if isinstance(data, pd.DataFrame):
            data = self.fill_dataset(
                data["dimuon_mass"].values, self.workspace.obj("mass"), ds_name=ds_name
            )
        elif isinstance(data, pd.Series):
            data = self.fill_dataset(
                data.values, self.workspace.obj("mass"), ds_name=ds_name
            )
        elif not (
            isinstance(data, rt.TH1F)
            or isinstance(data, rt.RooDataSet)
            or isinstance(data, rt.RooDataHist)
        ):
            raise Exception(f"Error: trying to add data of wrong type: {type(data)}")

        if blinded:
            data = data.reduce(rt.RooFit.CutRange("sideband_left,sideband_right"))

        self.data_registry[ds_name] = type(data)
        # self.workspace.Import(data, ds_name)
        getattr(self.workspace, "import")(data, ds_name)

    def fit_pseudodata(
        self,
        label="test",
        category="cat0",
        blinded=False,
        model_names=[],
        orders={},
        fix_parameters=False,
        title="",
        save=True,
        save_path="./",
        norm=0,
    ):
        tag = f"_{self.channel}_{category}"
        chi2 = {}
        model_names_all = []
        for model_name in model_names:
            if (model_name in self.requires_order) and (model_name in orders.keys()):
                for order in orders[model_name]:
                    model_names_all.append({"name": model_name, "order": order})
            else:
                model_names_all.append({"name": model_name, "order": 0})

        for model_names_order in model_names_all:
            model_name = model_names_order["name"]
            order = model_names_order["order"]
            if model_name in self.requires_order:
                model_key = f"{model_name}{order}" + tag
            else:
                model_key = model_name + tag
            # print(model_key)
            # self.workspace.pdf(model_key).Print()
            data = self.workspace.pdf(model_key).generate(
                rt.RooArgSet(self.workspace.obj("mass")), norm
            )
            ds_name = f"pseudodata_{model_key}"
            self.add_data(data, ds_name=ds_name)
            chi2[model_key] = self.fit(
                ds_name,
                norm,
                [model_name],
                orders={model_name: order},
                blinded=blinded,
                fix_parameters=fix_parameters,
                category=category,
                label=label,
                title=title,
                save=save,
                save_path=save_path,
                norm=norm,
            )[model_key]
        if save:
            mkdir(save_path)
            self.save_workspace(
                f"{save_path}/workspace_{self.channel}_{category}_{label}{self.filename_ext}"
            )
        return chi2

    def fill_dataset(self, data, x, ds_name="ds"):
        cols = rt.RooArgSet(x)
        ds = rt.RooDataSet(ds_name, ds_name, cols)
        for datum in data:
            if (datum < x.getMax()) and (datum > x.getMin()):
                x.setVal(datum)
                ds.add(cols)
        return ds

    def generate_data(self, model_name, category, xSec, lumi):
        tag = f"_{self.channel}_{category}"
        model_key = model_name + tag
        if model_key not in self.model_registry:
            self.add_model(model_name, category=category)
        return self.workspace.pdf(model_key).generate(
            rt.RooArgSet(self.workspace.obj("mass")), xSec * lumi
        )

    def add_model(self, model_name, order=None, category="cat0", prefix=""):
        if model_name not in self.fitmodels.keys():
            raise Exception(f"Error: model {model_name} does not exist!")
        tag = f"_{self.channel}_{category}"
        if order is None:
            model, params = self.fitmodels[model_name](self.workspace.obj("mass"), tag)
        else:
            if model_name in self.requires_order:
                model, params = self.fitmodels[model_name](
                    self.workspace.obj("mass"), tag, order
                )
            else:
                raise Exception(
                    f"Warning: model {model_name} does not require to specify order!"
                )

        model_key = model_name + tag
        if model_key not in self.model_registry:
            self.model_registry.append(model_key)
        # self.workspace.Import(model)
        getattr(self.workspace, "import")(model)

    def CorePdfFit(
        self,
        dataset=None,
        label="test",
        category="cat0",
        blinded=False,
        model_names=[],
        orders={},
        fix_parameters=False,
        store_multipdf=False,
        title="",
        save=True,
        save_path="./",
        norm=0,
    ):
        corePDF_results = {}
        hists_All = {}
        nCats = 5
        coreModelNames = ["bwz_redux_model"]
        for fitmodel in coreModelNames:
            add_model(ws, fitmodel, "_" + processName + "_corepdf")
        fixparam = False
        isBlinded = False
        name = "fake_data_Background_corPdfFit" + args.ext
        title = "Background"
        dataStack = rt.THStack("full_data", "full_data")
        for icat in range(nCats):
            hist_name = "hist" + "_" + processName + "_cat" + str(icat)
            ds = self.generate_data(
                ws, "bwz_redux_model", "_" + processName + "_corepdf", 100, lumi
            )
            hist = rt.RooAbsData.createHistogram(
                ds, hist_name, ws.var("mass"), rt.RooFit.Binning(80)
            )
            # hist.Write()
            hists_All[hist_name] = hist
            print(hists_All)
            print(hists_All[hist_name].Integral())
            # fake_data.append(ds)
            dataStack.Add(hist)
            self.add_data(ws, hist, False, hist_name + "_fake", False)
        print(hists_All)
        dataStack_Full = dataStack.GetStack().Last()
        hists_All[dataStack_Full.GetName()] = dataStack_Full
        # dataStack_Full.Write()
        fullDataSet = rt.RooDataHist(
            "core_Data", "core_Data", rt.RooArgList(ws.var("mass")), dataStack_Full
        )
        self.add_data(ws, fullDataSet, False, "_Core_fake", False)
        ws.Print()
        # plotter(ws,["ds_fake"], False, category, "data_bwZreduxmodel","BWZRedux model fake Data")
        corepdf_chi2 = self.fit(
            ws,
            "ds_Core_fake",
            coreModelNames,
            isBlinded,
            fixparam,
            "_" + processName + "_corepdf",
            True,
            name,
            title,
        )
        norm_Core = rt.RooRealVar(
            "bkg_norm_Core",
            "bkg_norm_Core",
            fullDataSet.sumEntries(),
            -float("inf"),
            float("inf"),
        )
        for icat in range(nCats):
            histName = "hist" + "_" + processName + "_cat" + str(icat)
            ws_corepdf = rt.RooWorkspace("ws_corepdf", False)
            ws_corepdf.Import(
                ws.pdf("bwz_redux_model" + "_" + processName + "_corepdf")
            )
            prefix = "cat" + str(icat)
            print(hists_All)
            transferHist = hists_All[histName].Clone()
            transferHist.Divide(dataStack_Full)
            transferHist.Scale(1 / transferHist.Integral())
            transferDataSet = rt.RooDataHist(
                "transfer_" + prefix,
                "transfer_" + prefix,
                rt.RooArgList(ws.var("mass")),
                transferHist,
            )
            transferDataName = transferDataSet.GetName()
            ws.Import(transferDataSet)
            ws_corepdf.Import(transferDataSet)
            chebyOrder = 3 if icat < 1 else 2
            add_model(
                ws,
                "chebychev_" + str(chebyOrder) + "_model",
                "_" + processName + "_" + prefix,
            )
            # transferFuncName = "chebychev_"+str(chebyOrder)+"_"+processName+"_"+prefix
            transferFuncName = "chebychev" + str(chebyOrder)
            chi2 = self.fit(
                ws,
                transferDataName,
                [transferFuncName],
                isBlinded,
                fixparam,
                "_" + processName + "_" + prefix,
                True,
                name,
                title,
            )
            coreBWZRedux = rt.RooProdPdf(
                "bkg_bwzredux_" + "_" + processName + "_" + prefix,
                "bkg_bwzredux_" + "_" + processName + "_" + prefix,
                ws.pdf("bwz_redux_model" + "_" + processName + "_corepdf"),
                ws.pdf(transferFuncName + "_" + processName + "_" + prefix),
            )
            ws_corepdf.Import(coreBWZRedux, rt.RooFit.RecycleConflictNodes())
            cat_dataSet = rt.RooDataHist(
                "data_" + prefix,
                "data_" + prefix,
                rt.RooArgList(ws.var("mass")),
                transferDataSet,
            )
            ndata_cat = cat_dataSet.sumEntries()
            norm_cat = rt.RooRealVar(
                "bkg_" + prefix + "_pdf_norm",
                "bkg_" + prefix + "_pdf_norm",
                ndata_cat,
                -float("inf"),
                float("inf"),
            )
            ws_corepdf.Import(cat_dataSet)
            ws_corepdf.Import(norm_cat)
            ws_corepdf.Import(norm_Core)
        if save:
            mkdir(save_path)
            self.save_workspace(
                f"{save_path}/workspace_{self.channel}_{category}_{label}{self.filename_ext}"
            )
        return chi2

    def fit(
        self,
        ds_name,
        ndata,
        model_names,
        orders={},
        blinded=False,
        fix_parameters=False,
        save=False,
        save_path="./",
        category="cat0",
        label="",
        title="",
        norm=0,
    ):
        if ds_name not in self.data_registry.keys():
            raise Exception(f"Error: Dataset {ds_name} not in workspace!")

        pdfs = {}
        chi2 = {}
        tag = f"_{self.channel}_{category}"
        model_names_all = []
        for model_name in model_names:
            if (model_name in self.requires_order) and (model_name in orders.keys()):
                for order in orders[model_name]:
                    model_names_all.append(f"{model_name}{order}")
            else:
                model_names_all.append(model_name)
        for model_name in model_names_all:
            model_key = model_name + tag
            pdfs[model_key] = self.workspace.pdf(model_key)
            pdfs[model_key].fitTo(
                self.workspace.obj(ds_name),
                rt.RooFit.Save(),
                rt.RooFit.PrintLevel(-1),
                rt.RooFit.Verbose(rt.kFALSE),
            )
            if fix_parameters:
                pdfs[model_key].getParameters(rt.RooArgSet()).setAttribAll("Constant")
            chi2[model_key] = self.get_chi2(model_key, ds_name, ndata)

            norm_var = rt.RooRealVar(f"{model_key}_norm", f"{model_key}_norm", norm)
            try:
                # self.workspace.Import(norm_var)
                getattr(self.workspace, "import")(norm_var)
            except Exception:
                print(f"{norm_var} already exists in workspace, skipping...")

        if save:
            mkdir(save_path)
            plot(self, ds_name, pdfs, blinded, category, label, title, save_path)

        return chi2

    def get_chi2(self, model_key, ds_name, ndata):
        normalization = rt.RooRealVar(
            "normaliazation", "normalization", ndata, 0.5 * ndata, 2 * ndata
        )
        model = rt.RooExtendPdf(
            "ext", "ext", self.workspace.pdf(model_key), normalization
        )
        xframe = self.workspace.obj("mass").frame()
        ds = self.workspace.obj(ds_name)
        ds.plotOn(xframe, rt.RooFit.Name(ds_name))
        model.plotOn(xframe, rt.RooFit.Name(model_key))
        nparam = model.getParameters(ds).getSize()
        chi2 = xframe.chiSquare(model_key, ds_name, nparam)
        if chi2 <= 0:
            chi2 == 999
        return chi2
