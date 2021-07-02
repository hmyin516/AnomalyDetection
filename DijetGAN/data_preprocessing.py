import numpy as np
import pandas as pd
import gc

TESTING = False

SB_WIDTH = 2

filenames = {
    "herwig": "../data/events_anomalydetection_DelphesHerwig_qcd_features.h5",
    "pythiabg": "../data/events_anomalydetection_DelphesPythia8_v2_qcd_features.h5",
    "pythiasig": "../data/events_anomalydetection_DelphesPythia8_v2_Wprime_features.h5"
}

datatypes = ["herwig", "pythiabg", "pythiasig"]

train_features = ["ptj1", "etaj1", "mj1", "ptj2", "etaj2", "phij2", "mj2", "tau21j1", "tau21j2", "tau32j1", "tau32j2"]
condition_features = ["mjj"]

features = train_features + condition_features

def load_data(datatype, stop = None, rotate = True, flip_eta = True):
    input_frame = pd.read_hdf(filenames[datatype], stop = stop)
    output_frame = pd.DataFrame(dtype = "float32")

    for jet in ["j1", "j2"]:
        output_frame["pt" + jet] = np.sqrt(input_frame["px" + jet]**2 + input_frame["py" + jet]**2)
        output_frame["eta" + jet] = np.arcsinh(input_frame["pz" + jet] / output_frame["pt" + jet])
        output_frame["phi" + jet] = np.arctan2(input_frame["py" + jet], input_frame["px" + jet])
        output_frame["m" + jet] = input_frame["m" + jet]
        output_frame["p" + jet] = np.sqrt(input_frame["pz" + jet]**2 + output_frame["pt" + jet]**2)
        output_frame["e" + jet] = np.sqrt(output_frame["m" + jet]**2 + output_frame["p" + jet]**2)
        output_frame["tau21" + jet] = input_frame["tau2" + jet] / input_frame["tau1" + jet]
        output_frame["tau32" + jet] = input_frame["tau3" + jet] / input_frame["tau2" + jet]
    
    del input_frame
    gc.collect()

    # Not exact rotation, since negative angles for phi2 are flipped across the x-axis. Should be OK due to symmetry.
    if rotate:
        output_frame["phij2"] = np.abs(output_frame["phij2"] - output_frame["phij1"])
        output_frame["phij1"] = 0
    
    if flip_eta:
        flipped_frame = output_frame.copy()
        flipped_frame["etaj1"] *= -1
        flipped_frame["etaj2"] *= -1
        output_frame = output_frame.append(flipped_frame)
        del flipped_frame
        gc.collect()
    
    for jet in ["j1", "j2"]:
        output_frame["px" + jet] = output_frame["pt" + jet] * np.cos(output_frame["phi" + jet])
        output_frame["py" + jet] = output_frame["pt" + jet] * np.sin(output_frame["phi" + jet])
        output_frame["pz" + jet] = output_frame["pt" + jet] * np.sinh(output_frame["eta" + jet])
    
    # Dijet properties
    output_frame["pxjj"] = output_frame["pxj1"] + output_frame["pxj2"]
    output_frame["pyjj"] = output_frame["pyj1"] + output_frame["pyj2"]
    output_frame["pzjj"] = output_frame["pzj1"] + output_frame["pzj2"]
    output_frame["ejj"] = output_frame["ej1"] + output_frame["ej2"]
    output_frame["pjj"] = np.sqrt(output_frame["pxjj"]**2 + output_frame["pyjj"]**2 + output_frame["pzjj"]**2)
    output_frame["mjj"] = np.sqrt(output_frame["ejj"]**2 - output_frame["pjj"]**2)

    # NaNs may arise from overly sparse jets with tau1 = 0, tau2 = 0, etc.
    output_frame.dropna(inplace = True)
    output_frame.reset_index(drop = True, inplace = True)
    
    return output_frame.astype('float32')

if TESTING:
    df_bg = load_data("pythiabg", stop = 10000)
    df_sig = load_data("pythiasig", stop = 1000)
else:
    df_bg = load_data("pythiabg")
    df_sig = load_data("pythiasig")

# Define sidebands and signal region

sr_center = 3500
sr_width = 250

sr_left = sr_center - sr_width
sr_right = sr_center + sr_width

sb_left = sr_left - sr_width * SB_WIDTH
sb_right = sr_right + sr_width * SB_WIDTH

df_bg_SB = df_bg[((df_bg["mjj"] > sb_left) & (df_bg["mjj"] < sr_left)) | ((df_bg["mjj"] > sr_right) & (df_bg["mjj"] < sb_right))]
df_bg_SR = df_bg[(df_bg["mjj"] >= sr_left) & (df_bg["mjj"] <= sr_right)]

df_sig_SB = df_sig[((df_sig["mjj"] > sb_left) & (df_sig["mjj"] < sr_left)) | ((df_sig["mjj"] > sr_right) & (df_sig["mjj"] < sb_right))] # This should pretty much be empty
df_sig_SR = df_sig[(df_sig["mjj"] >= sr_left) & (df_sig["mjj"] <= sr_right)]

for df in [df_bg_SB, df_bg_SR, df_sig_SB, df_sig_SR]:
    df.reset_index(drop = True, inplace = True)

print("Size of uncut data:")
print("df_bg shape {}".format(df_bg.shape))
print("df_sig shape {}".format(df_sig.shape))
print("df_bg_SB shape {}".format(df_bg_SB.shape))
print("df_bg_SR shape {}".format(df_bg_SR.shape))
print("df_sig_SB shape {}".format(df_sig_SB.shape))
print("df_sig_SR shape {}".format(df_sig_SR.shape))
print()

def cut_data(uncut_data, pTmin = 1200, etamax = 2.5):
    # Column 0: ptj1
    # Column 1: etaj1
    # Column 3: ptj2
    # Column 4: etaj2
    return uncut_data[((uncut_data[:,0] > pTmin) & (np.abs(uncut_data[:,1]) < etamax)) | ((uncut_data[:,3] > pTmin) & (np.abs(uncut_data[:,4]) < etamax))]

np_bg_SB = cut_data(np.array(df_bg_SB[features]))
np_bg_SR = cut_data(np.array(df_bg_SR[features]))
np_sig_SB = cut_data(np.array(df_sig_SB[features]))
np_sig_SR = cut_data(np.array(df_sig_SR[features]))

del df_bg_SB
del df_bg_SR
del df_sig_SB
del df_sig_SR
gc.collect()

np.save("../data/processed-500SR/np_bg_SB_" + str(SB_WIDTH) + ".npy", np_bg_SB)
np.save("../data/processed-500SR/np_bg_SR_" + str(SB_WIDTH) + ".npy", np_bg_SR)
np.save("../data/processed-500SR/np_sig_SB_" + str(SB_WIDTH) + ".npy", np_sig_SB)
np.save("../data/processed-500SR/np_sig_SR_" + str(SB_WIDTH) + ".npy", np_sig_SR)

print("Size of cut data:")
print("np_bg_SB shape {}".format(np_bg_SB.shape))
print("np_bg_SR shape {}".format(np_bg_SR.shape))
print("np_sig_SB shape {}".format(np_sig_SB.shape))
print("np_sig_SR shape {}".format(np_sig_SR.shape))
print()