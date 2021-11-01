from typing import no_type_check
from mc_sim.imports import *
import mc_sim.models as models
import mc_sim.MCMC as MCMC
import mc_sim.multiproc as multiproc
from mpi4py.MPI import ANY_SOURCE

#Constants
sigma_parameter=0.024
column_names=['v','t_sky']
freq_central=75

#Parameters required for metropolis algorithm

num_walkers=4
num_samples=100000
thining=3
test_to_run ="test_2"#used to name saved data  
#np.random.seed(1)



def set_multiproc():
    global multiprocessing_obj
    multiprocessing_obj= multiproc.multiproc()

    global rank
    rank = multiprocessing_obj.getRank(multiprocessing_obj.comm)

    global size_cores
    size_cores = multiprocessing_obj.getSize(multiprocessing_obj.comm)

    global sampleBy_processor
    sampleBy_processor = int(num_samples/size_cores)

    

def run_Metropolis():
    #Running Metropolis algorithm
    dataframes = Metropolis.MH(sky_model,
                                    parameters,
                                    t_sky_data,
                                    sigma_parameter,
                                    evaluateLogLikelihood,
                                    initial_point,
                                    num_walkers,
                                    rank,
                                    multiprocessing_obj.comm,
                                    size_cores,
                                    sampleBy_processor)   
    if rank==0:
        mergedData=chainMerge(dataframes)
        dataProcessing(mergedData)


def chainMerge(dataframes):

    df = pd.DataFrame(columns=dataframes[0].columns)

    for i in range(sampleBy_processor):
        for data in dataframes:
           df= df.append(data.loc[i],ignore_index=True)

    return df

    pass

def dataProcessing(dataframe):
    total_stack = Metropolis.thining(dataframe,thining,sampleBy_processor )

    print(total_stack)
    
    sec = saving(total_stack)
    plot_distributions(total_stack,sec)

def plot_distributions(total_stack,sec):

    total_stack = total_stack[['b0',"b1","b2","b3","Te"]]
    print("processing Pair plot KDE....")
    fig= sns.pairplot(total_stack, kind="kde",diag_kws = {'alpha':0.8})
    fig.savefig(test_to_run+'/pair_plot_KDE_'+str(sec)+'.png')

def saving(total_stack):
    if not os.path.exists(test_to_run):
        os.makedirs(test_to_run)

    sec = secrets.token_hex(nbytes=8)
    sns.set_theme()
    plt.figure()

    plot = sns.relplot(
        data=total_stack[["iteration","likelihood","walker","accepted"]],kind="line",
        x="iteration", y="likelihood",
        hue="accepted",facet_kws=dict(sharex=False),
    )

    """Saving Results"""
    plot.figure.savefig(test_to_run+"/"+str(sec)+"_dash.png") 
    total_stack.to_csv(test_to_run+"/"+str(sec)+'.csv', index=False)

    return sec

def model_evaluation(theta,v):
    return sky_model.EVAL_t_sky_model_full(theta,v,freq_central=75)
   

def evaluateLogLikelihood(theta,v,y,sigma):
    log_likelihood = 0
    yerr = 0.02 + 0.008* np.random.uniform(0,1,1) #to consider an error in sigma calculation
    sn2 = yerr[0] ** 2
    #sn2=sigma**2  #this is used as the study case requires it this way
                    #Next step is to count sigma squared as a parameter to be tested as well

    evaluated_model = model_evaluation(theta,v)

    log_likelihood = ((-((1/(2*sn2))*sum(np.square(y-evaluated_model))) - (0.5*len(y)*np.log(2*np.pi*sn2))))

    return log_likelihood



def main():
    global sky_model 
    sky_model=models.Sky_model()

    global parameters 
    parameters=sky_model.init_params()

    global Metropolis 
    Metropolis = MCMC.MCMC()

    global initial_point
    initial_point = Metropolis.gaussian_sample(parameters,1)

    global t_sky_data
    t_sky_data= sky_model.load_data()

    set_multiproc()
    run_Metropolis()



if __name__ == "__main__":
    main()