"""
Author: Jorge H. Cárdenas
Author: Germán Chaparro

University of Antioquia

"""

from mc_sim.imports import *


class Sky_model:
    
    parameters={}
    freq_central=0
    x_values=[]
    
    def __init__(self):
        self.parameters = {}
        
    def load_data(self):
        measure = pd.read_csv('figure1_plotdata.csv') 
        measure.columns
        measure = measure[measure[" a: Tsky [K]"] != 0]

        t_sky_data = measure[['Frequency [MHz]',' a: Tsky [K]']]
        t_sky_data = t_sky_data.rename(columns={'Frequency [MHz]': 'Freq', ' a: Tsky [K]': 't_sky'})


        return t_sky_data
    
    def init_params(self):
        parameters={
            "b0": {
                "initial":1755,
                "min":1700,
                "max":1760,
                "nu" : 1730,
                "sigma" : 20
            },
            "b1": {
                "initial":-0.08,
                "min":-0.1,
                "max":-0.07,
                "nu" : -0.08,
                "sigma" : 0.01
            },
            "b2": {
                "initial":-0.012,
                "min":-0.1,
                "max":0.011,
                "nu" : -0.012,
                "sigma" : 0.01
            },
            "b3": {
                "initial":0.0052,
                "min":0.0010,
                "max":0.2,
                "nu" : 0.0052,
                "sigma" : 0.01
            },
            "Te":{
                "initial":800,
                "min":200,
                "max":2000,
                "nu" : 500,
                "sigma" : 10
            },
        }
        
        return parameters
    
    def update_parameters(self,values):
        parameters={
            "b0": {
                "initial":1755,
                "min":1700,
                "max":1760,
                "nu" : values[0],
                "sigma" : 50
            },
            "b1": {
                "initial":-0.08,
                "min":-0.1,
                "max":-0.07,
                "nu" : values[1],
                "sigma" : 0.001
            },
            "b2": {
                "initial":-0.012,
                "min":-0.1,
                "max":0.011,
                "nu" : values[2],
                "sigma" : 0.001
            },
            "b3": {
                "initial":0.0052,
                "min":0.0010,
                "max":0.2,
                "nu" : values[3],
                "sigma" : 0.001
            },
            "Te":{
                "initial":800,
                "min":200,
                "max":2000,
                "nu" : values[4],
                "sigma" : 10
        
            },
        }
        return parameters
    
    def t_sky_model_full(self,parameters,x,freq_central):
        self.parameters=parameters
        self.x_values=x
        self.freq_central=freq_central
        v=x
        vc=freq_central
        b0=parameters["b0"]["initial"]
        b1=parameters["b1"]["initial"]
        b2=parameters["b2"]["initial"]
        b3=parameters["b3"]["initial"]
        Te=parameters["Te"]["initial"]
        
        rate=(v/vc)**-2.0
        
        return (b0*(v/vc)**(-2.5+b1+b2*np.log10(v/vc)))*np.exp(-b3*rate)+Te*(1-np.exp(-b3*rate))
    
    def EVAL_t_sky_model_full(self,parameters,x,freq_central):
        self.parameters=parameters
        self.x_values=x
        self.freq_central=freq_central
        v=x
        vc=freq_central
        b0=parameters[0]
        b1=parameters[1]
        b2=parameters[2]
        b3=parameters[3]
        Te=parameters[4]

        rate=(v/vc)**-2.0
        
        return (b0*(v/vc)**(-2.5+b1+b2*np.log10(v/vc)))*np.exp(-b3*rate)+Te*(1-np.exp(-b3*rate))
    
    def t_sky_model_full_b0_b3(self,parameters,x,freq_central):
        self.parameters=parameters
        self.x_values=x
        self.freq_central=freq_central
        v=x
        vc=freq_central
        b0=parameters["b0"]["initial"]
        b3=parameters["b3"]["initial"]
        
        rate=(v/vc)**-2.0
        
        return (b0*(v/vc)**(-2.5))*np.exp(-b3*rate)

    def EVAL_t_sky_model_full_b0_b3(self,parameters,x,freq_central):
        self.parameters=parameters
        self.x_values=x
        self.freq_central=freq_central
        v=x
        vc=freq_central
        b0=parameters[0]
        b3=parameters[1]
        
        rate=(v/vc)**-2.0
        
        return (b0*(v/vc)**(-2.5))*np.exp(-b3*rate)
    
    

    def t_sky_model_b0(self,parameters,x,freq_central):
        
        self.parameters=parameters
        self.x_values=x
        self.freq_central=freq_central
        v=x
        vc=freq_central
        b0=parameters["b0"]["initial"]

        
        rate=(v/vc)**-2.0
        
        return (b0*(v/vc)**(-2.5))
    
    def t_sky_model_No_Te(self,parameters,x,freq_central):
        
        self.parameters=parameters
        self.x_values=x
        self.freq_central=freq_central
        v=x
        vc=freq_central
        b0=parameters["b0"]["initial"]
        b1=parameters["b1"]["initial"]
        b2=parameters["b2"]["initial"]
        b3=parameters["b3"]["initial"]
        
        rate=(v/vc)**-2.0
        
        return (b0*(v/vc)**(-2.5+b1+b2*np.log10(v/vc)))*np.exp(-b3*rate)

    def t_Te(self,parameters,x,freq_central):
        
        self.parameters=parameters
        self.x_values=x
        self.freq_central=freq_central
        v=x
        vc=freq_central
        b3=parameters["b3"]["initial"]
        Te=parameters["Te"]["initial"]
        
        rate=(v/vc)**-2.0

        return Te*(1-np.exp(-b3*rate))