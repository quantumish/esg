import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from zope.interface import *
import distinctipy
import random
import time

random.seed(time.time())

@dataclass
class Plant:
    name: str
    capacity: int
    var_cost: float
    daily_om: float
    price: Optional[float]
    used: float

class Strategy(Interface):
    def set_prices(portfolios, hour, plants: List[Plant]) -> None:
        """Sets prices of power plants."""

@dataclass
class Portfolio:
    name: str    
    plants: List[Plant]
    strategy: Optional[Strategy]
    money: float
    
    def net_operating_costs(self):
        return sum([
            (plant.capacity * plant.var_cost) + plant.daily_om
            for plant in self.plants
        ])

    def total_prod(self):
        return sum([plant.capacity for plant in self.plants])
    
    def set_prices(self, portfolios, hour):
        self.strategy.set_prices(portfolios, hour, self.plants)

    def calc_gain(self, eod=False):
        for plant in self.plants:
            self.money += plant.used * plant.price
            self.money -= plant.used * plant.var_cost
            if eod:
                self.money -= plant.daily_om         
        if eod:
            self.money *= 1.05 # HACK?

    def __eq__(self, other):
        return self.money == other.money
    def __lt__(self, other):
        return self.money < other.money        

            
@implementer(Strategy)
class NaiveBreakeven:
    def set_prices(portfolios, hour, plants):
        for plant in plants:
            plant.price = plant.var_cost


@implementer(Strategy)
class SlightlyLessNaiveBreakeven:
    """Aims for breakeven but factors in the O&M cost.
    Doesn't account for the fact that not all stations turn on, though.
    """
    def set_prices(portfolios, hour, plants):
        for plant in plants:
            plant.price = plant.var_cost + (plant.daily_om/(4*plant.capacity))

@implementer(Strategy)
class SlightProfit:
    """Tries to make a slight profit via a 10% markup."""
    def set_prices(portfolios, hour, plants):
        for plant in plants:
            plant.price = plant.var_cost * 1.10
@dataclass
class Price:
    src: Plant
    asker: str
    capacity: int
    value: float
    
    def __eq__(self, other):
        return self.value == other.value
    def __lt__(self, other):
        return self.value < other.value        

def aggregate_prices(portfolios):
    prices = []    
    for p in portfolios:
        for plant in p.plants:
            prices.append(Price(plant, p.name, plant.capacity, plant.price))
    return sorted(prices)

def get_step_representation(prices):
    steps = []
    totalx = 0
    # totaly = 0
    for price in prices:
        steps.append((totalx, price.value, price))
        totalx+=price.capacity
        # totaly+=price.value
    return steps

def plot_prices(prices, pallette):    
    totalx = 0
    oldy = 0
    plt.ylabel("Asking price")
    plt.xlabel("MWh")
    for price in prices:
        plt.step([totalx, totalx+price.capacity], [oldy, price.value], color=pallette[price.asker])
        totalx+=price.capacity
        oldy=price.value
        #totaly+=price.value

class Demand:
    def __init__(self, hour):        
        demands = pd.read_csv("./demands.csv")
        self.x_it = demands["Demand at 0"][hour-1]
        self.y_it = abs(demands["Demand at 0"][hour-1]*demands["Slope"][hour-1])
        self.slope = demands["Slope"][hour-1]
        self.f = lambda x: self.y_it + self.slope*x
    
def plot_demand_curve(hour):
    plt.title(f"Hour {hour}")
    demands = pd.read_csv("./demands.csv")
    # slope = 1/demands["Slope"][hour]
    x_it = demands["Demand at 0"][hour-1]
    y_it = abs(demands["Demand at 0"][hour-1]*demands["Slope"][hour-1])
    xs = np.linspace(0, x_it, 100)
    plt.plot(xs, [y_it + demands["Slope"][hour-1]*x for x in xs], color="gray")

def plot_hour(portfolios, prices, hour):
    colors = distinctipy.get_colormap(distinctipy.get_colors(len(portfolios)))
    pallette = {}
    for i, p in enumerate(portfolios):
        pallette[p.name] = colors.colors[i]

    plot_prices(prices, pallette)
        
    # for i in range(16):
    plot_demand_curve(hour)
    plt.ylim([0, max([price.value for price in prices])*1.01])
    plt.xlim([0, sum([p.total_prod() for p in portfolios])*1.01])
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=pallette[portfolios[i].name], lw=4) for i in range(len(portfolios))]
    plt.legend(custom_lines, [p.name for p in portfolios])
    
    plt.show()

@implementer(Strategy)
class ConservativePrePlan:
    """Assumes everyone else is playing DEFAULT_STRAT, then tries to maximize price of
    powerplants without upsetting the aggreggate supply function, thus earning more."""
    # HACK just so hacky
    def set_prices(portfolios, hour, plants):
        copy = portfolios.copy()
        prices = []
        for p in copy:
            DEFAULT_STRAT.set_prices(portfolios, hour, p.plants) 
            for plant in p.plants:
                prices.append(Price(plant, p.name, plant.capacity, plant.price))
        prices.sort()
        for i, price in enumerate(prices):
            if price.src in plants and i+1 < len(prices):
                plant = plants[plants.index(price.src)]                
                next_price = prices[i+1]
                counter = 1
                while i+counter < len(prices)-1:                    
                    plant.price = max(prices[i+counter].value-0.001, 0)
                    if next_price.src not in plants:
                        break
                    counter += 1
                    next_price = prices[i+counter]
        

DEFAULT_STRAT = SlightProfit # NaiveBreakeven
            
def get_portfolios():
    portfolios = []
    for f in os.listdir("./portfolios"):
        df = pd.read_csv(f"./portfolios/{f}")
        portfolio = Portfolio(f[:-4], [], random.choice([NaiveBreakeven, SlightProfit, SlightlyLessNaiveBreakeven, ConservativePrePlan]), 0)
        # print(portfolio.strategy)
        for i, row in df.iterrows():
            portfolio.plants.append(Plant(
                name=row["UNITNAME"].lower(),
                capacity=row["Capacity"],
                var_cost=row["TotalVarCost"],
                daily_om=row["O&M/Day"],
                price=None,
                used=0
            ))
        portfolios.append(portfolio)
    
    return portfolios      

def run_sim(portfolios):
    for hour in range(1,17):
        for p in portfolios:
            p.set_prices(portfolios, hour)
        prices = aggregate_prices(portfolios)
        steps = get_step_representation(prices)

        demand = Demand(hour)
        for step in steps:
            d = demand.f(step[0]+step[2].capacity)
            if step[1] > d:        
                step[2].src.used = step[2].capacity - ((step[1]-d)/demand.slope)
                break
            else:
                step[2].src.used = step[2].capacity

        # print(f"HOUR {hour}")
        for p in portfolios:        
            p.calc_gain(eod=(hour % 4 == 0))

        plot_hour(portfolios, prices, hour)         
                    
# def get_net_gain(agg_prices, portfolio, hour):  

for i in range(1):
    portfolios = get_portfolios()    
    portfolios[i].strategy = ConservativePrePlan
    # plot_hour(portfolios, prices, 0)
    
    print("\n", portfolios[i].name, "\n")
    run_sim(portfolios)
    
    for p in reversed(sorted(portfolios)):
        print(p.name, p.money)
        # print(hour)    
        # plot_hour(portfolios, prices, hour)    
        
