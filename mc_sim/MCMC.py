"""
Author: Jorge H. Cárdenas
University of Antioquia

"""

from mc_sim.imports import *

import math

class MCMC:
    
    output_data = 0
    
    def __init__(self):
        pass
    

        
    def gaussian_sample(self,params,N):

        theta_sampled=[]
        for key, value in params.items():

            selected_samples = np.random.normal(value["nu"], value["sigma"], N)
            theta_sampled.append(selected_samples[0])

        return theta_sampled

    def mirror_gaussian_sample(self,params,N):

        theta_sampled=[]
        intermediate_sample=[]
        
        for key, value in params.items():

            prior_eval=-np.inf
            
            while prior_eval==-np.inf:
                
                #selected_samples = np.random.uniform(low=value["min"], high=value["max"], size=1)

                selected_samples = np.random.normal(value["nu"], value["sigma"], 1*N)
                prior_eval = self.prior(selected_samples,value["min"],value["max"])

            
            intermediate_sample.append(selected_samples)

        
        return np.mean(np.array(intermediate_sample).T,axis=0)
        #return np.array(intermediate_sample).T

#Assign values based on current state
#Get an array with the evaluation for an specific parameter value in the whole range of X
    
   

    def prior(self,theta,minm, maxm):
        #in this case priors are just the required check of parameter conditions
        #it is unknown.
        #it must return an array with prior evaluation of every theta
        #i evaluate a SINGLE THETA, A SINGLE PARAMETER EVERY TIME
        #depending on which conditional probability i am evaluating

        #m, b, log_f = theta
        if minm < theta < maxm:
            return True
        #return -np.inf
        return True
    #just assuming everything is in range

    # Metropolis-Hastings

    def set_dataframe(self,parameters):
        columns=['iteration','walker','accepted','likelihood']

        global_simulation_data = pd.DataFrame(columns=columns)  
        new_dtypes = {"iteration": int,"likelihood":np.float64,"walker":int,"accepted":"bool"}
        global_simulation_data[columns] = global_simulation_data[columns].astype(new_dtypes)


        for parameter in parameters:
            global_simulation_data.insert(len(global_simulation_data.columns), parameter,'' )
            new_dtypes = {parameter:np.float64}
            global_simulation_data[parameter] = global_simulation_data[parameter].astype(np.float64)

        return global_simulation_data

    def acceptance(self,new_loglik,old_log_lik):
        if (new_loglik > old_log_lik):

            return True
        else:
            u = np.random.uniform(0.0,1.0)
            # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
            # less likely x_new are less likely to be accepted
            return (u < (np.exp(new_loglik - old_log_lik)))

    def thining(self,dataframe,thining,num_samples ):
        stack = pd.DataFrame()
        walkers = dataframe.walker.unique()
        
        for walker in walkers:

            selected = dataframe.loc[dataframe['walker'].isin([walker])]
            
            #selected =  selected.sample(frac=0.8)
            selected = selected.nsmallest(int(num_samples*0.55),['likelihood']) #Thining

            selected = selected[selected.index % thining == 0]

            stack = pd.concat([stack,selected],ignore_index=True)

        return stack.sort_values(by=['iteration']).reset_index(drop=True)
    

    def MH(self,sky_model,parameters,t_sky_data,sigma_parameter,evaluateLogLikelihood,initial_point,num_walkers,rank,comm,size_cores,num_samples):
        
        accepted  = 0.0
        row=0
        iteration=0
        thetas_samples=[]

        thetas_samples.append(initial_point)

        num_of_params=len(initial_point)
        walkers_result=[]
        dataframe_array = [self.set_dataframe(parameters) for i in range(size_cores)]

        if rank==0:

            with tqdm(total=(num_samples)) as pbar:
                    
                initial_point = self.gaussian_sample(parameters,1)

                for n in range(num_samples):

                    pbar.update(1)        

                    old_theta = np.array(thetas_samples[len(thetas_samples)-1], copy=True) 
                    old_log_lik = evaluateLogLikelihood(old_theta,t_sky_data.Freq,t_sky_data.t_sky,sigma_parameter)

                    params = sky_model.update_parameters(old_theta) #this has impact when using gaussian proposed distribution

                    new_theta = self.mirror_gaussian_sample(params,1)
                    new_loglik = evaluateLogLikelihood(new_theta,t_sky_data.Freq,t_sky_data.t_sky,sigma_parameter)

                    # Accept new candidate in Monte-Carlo.
                    if self.acceptance(new_loglik,old_log_lik):
                        thetas_samples.append(new_theta)
                        accepted = accepted + 1.0  # monitor acceptance

                        data = np.concatenate(([iteration, rank,1, new_loglik],new_theta),axis=0)
                        dataframe_array[rank].loc[iteration] = data

                    else:
                        thetas_samples.append(old_theta)
                        data = np.concatenate(([iteration, rank, 0, old_log_lik],old_theta),axis=0)

                        dataframe_array[rank].loc[iteration] = data

                    
                    for i in range(1, size_cores):

                        req2 = comm.irecv(source=i, tag=12)
                        received = req2.wait()
                        dataframe_array[i].loc[received[0]] = received



                    iteration += 1
        
            print("accepted"+str(accepted))                
            return dataframe_array
        else:
            with tqdm(total=(num_samples)) as pbar:
                    
                initial_point = self.gaussian_sample(parameters,1)

                for n in range(num_samples):
                    pbar.update(1)        

                    old_theta = np.array(thetas_samples[len(thetas_samples)-1], copy=True) 
                    old_log_lik = evaluateLogLikelihood(old_theta,t_sky_data.Freq,t_sky_data.t_sky,sigma_parameter)

                    params = sky_model.update_parameters(old_theta) #this has impact when using gaussian proposed distribution

                    new_theta = self.mirror_gaussian_sample(params,1)
                    new_loglik = evaluateLogLikelihood(new_theta,t_sky_data.Freq,t_sky_data.t_sky,sigma_parameter)

                    # Accept new candidate in Monte-Carlo.
                    if self.acceptance(new_loglik,old_log_lik):
                        thetas_samples.append(new_theta)
                        accepted = accepted + 1.0  # monitor acceptance

                        data = np.concatenate(([iteration, rank,1, new_loglik],new_theta),axis=0)
                        req = comm.isend(data, dest=0, tag=12)
                        req.wait()

                    else:
                        thetas_samples.append(old_theta)
                        data = np.concatenate(([iteration, rank, 0, old_log_lik],old_theta),axis=0)

                        req = comm.isend(data, dest=0, tag=12)
                        req.wait()

                    iteration += 1
            print("accepted"+str(accepted))                
            return np.inf
                



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # accepted  = 0.0
        # row=0
        # iteration=0
        # thetas_samples=[]

        # thetas_samples.append(initial_point)

        # num_of_params=len(initial_point)
        # walkers_result=[]
        # dataframe = self.set_dataframe(parameters)

        # #this division of rank 0 and others can be helpful to
        # #stablish a main node to manage the others

        # num_walkers_min = (int(num_walkers/size_cores) * rank )
        # num_walkers_max = num_walkers_min + int(num_walkers/size_cores) -1

        # print("\nwalkers to process" + str(num_walkers_max-num_walkers_min+1))
     

        # with tqdm(total=(num_samples*(num_walkers_max+1-num_walkers_min))) as pbar:

        #     for walker in range(num_walkers_min, num_walkers_max+1):
        #         iteration = num_samples*walker
        #     #initial_point = self.gaussian_sample(parameters,1)
        #         for n in range(num_samples):

        #             pbar.update(1)        

        #             old_theta = np.array(thetas_samples[len(thetas_samples)-1], copy=True) 
        #             old_log_lik = evaluateLogLikelihood(old_theta,t_sky_data.Freq,t_sky_data.t_sky,sigma_parameter)

        #             params = sky_model.update_parameters(old_theta) #this has impact when using gaussian proposed distribution

        #             new_theta = self.mirror_gaussian_sample(params,1)                        
                    
        #             new_loglik = evaluateLogLikelihood(new_theta,t_sky_data.Freq,t_sky_data.t_sky,sigma_parameter)

        #             # Accept new candidate in Monte-Carlo.
        #             if n>burn_sample:
        #                 if self.acceptance(new_loglik,old_log_lik):
        #                     accepted = accepted + 1.0  # monitor acceptance

        #                     if rank == 0:
        #                         data = np.concatenate(([iteration, walker,1, new_loglik],new_theta),axis=0)
        #                         dataframe.loc[iteration] = data

        #                         for i in range(1, size_cores):
        #                             data_req = np.empty(9, dtype = float)
                                    
        #                             #comm.Recv(data_req, source=i, tag=11)

        #                             req = comm.irecv(source=i, tag=11)
        #                             data_req = req.wait()
        #                             dataframe.loc[int(data_req[0])] = data_req

                                
        #                     else:

        #                         data = np.concatenate(([iteration, walker,1, new_loglik],new_theta),axis=0)
                                
        #                         #comm.Send(data, dest=0, tag=11)

        #                         req = comm.isend(data, dest=0, tag=11)
        #                         req.wait()

        #                 else:
                            

        #                     if rank == 0:
        #                         data = np.concatenate(([iteration, walker, 0, old_log_lik],old_theta),axis=0)

        #                         dataframe.loc[iteration] = data

        #                         for i in range(1, size_cores):
        #                             req = comm.irecv(source=i, tag=11)
        #                             data_req = req.wait()
        #                             #data_req = np.empty(9, dtype = float)
        #                             #comm.Recv(data_req, source=i, tag=11)

        #                             dataframe.loc[int(data_req[0])] = data_req

                                
        #                     else:
        #                         data = np.concatenate(([iteration, walker, 0, old_log_lik],old_theta),axis=0)
                                
        #                         #comm.Send(data, dest=0, tag=11)

        #                         req = comm.isend(data, dest=0, tag=11)
        #                         req.wait()

        #             iteration = iteration + 1
                    

        # return  dataframe

        
    

