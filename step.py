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
import math

order= [
    "big_coal",
    "big_gas",
    "bay_views",
    "beachfront",
    "east_bay",
    "old_timers",
    "fossil_light",
]

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
    def set_prices(self, portfolios, hour, plants: List[Plant]) -> None:
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

    def calc_gain(self, eod=False, bod=False):
        old = self.money
        for plant in self.plants:
            # if self.name == "fossil_light":
            #     print(plant.name, plant.used, plant.capacity, plant.price)
            self.money += plant.used * plant.price
            self.money -= plant.used * plant.var_cost
            if eod:
                self.money -= plant.daily_om
            plant.used = 0 # reset
        # print(f"expect {self.name} profit {self.money-old}")
        if bod:
            self.money *= 1.05 # HACK?
        # if self.name == "fossil_light":
        #     print(self.money)

    def __eq__(self, other):
        return self.money == other.money
    def __lt__(self, other):
        return self.money < other.money


@implementer(Strategy)
class NaiveBreakeven:
    def __init__(self, amt):
        self.amt = amt

    def set_prices(self, portfolios, hour, plants):
        for plant in plants:
            plant.price = plant.var_cost + self.amt

@implementer(Strategy)
class Asshole:
    def set_prices(self, portfolios, hour, plants):
        for plant in plants:
            plant.price = 0


@implementer(Strategy)
class SlightlyLessNaiveBreakeven:
    """Aims for breakeven but factors in the O&M cost.
    Doesn't account for the fact that not all stations turn on, though.
    """
    def set_prices(self, portfolios, hour, plants):
        for plant in plants:
            plant.price = plant.var_cost + (plant.daily_om/(4*plant.capacity))

@implementer(Strategy)
class PessimisticBreakeven:
    def set_prices(self, portfolios, hour, plants):
        def compare(p):
            return p.var_cost

        total_unaccounted_om = 0
        sort = sorted(plants, key=compare)

        half = math.floor(len(sort)/2)
        for i in range(half):
            total_unaccounted_om = sort[half+i].daily_om

        for plant in plants:
            plant.price = plant.var_cost + (total_unaccounted_om/(0.5*len(plants)*plant.capacity))

@implementer(Strategy)
class Intersection:
    def set_prices(self, portfolios, hour, plants):
        for plant in plants:
            plant.price = max(40, plant.var_cost)

@implementer(Strategy)
class SlightProfit:
    """Tries to make a slight profit via a 10% markup."""
    def set_prices(self, portfolios, hour, plants):
        for plant in plants:
            plant.price = plant.var_cost * 1.10

@dataclass
class Price:
    src: Plant
    asker: str
    capacity: int
    value: float

    def __eq__(self, other):
        return (self.value == other.value and self.asker == other.asker)
    def __lt__(self, other):
        if self.value == other.value:
             return order.index(self.asker) < order.index(other.asker)
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

    # for i in range(1,17):
    #     plot_demand_curve(i)
    plot_demand_curve(hour)

    plt.ylim([0, max([price.value for price in prices])*1.01])
    # plt.xlim([0, sum([p.total_prod() for p in portfolios])*1.01])

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=pallette[portfolios[i].name], lw=4) for i in range(len(portfolios))]
    plt.legend(custom_lines, [p.name for p in portfolios])

    plt.show()

@implementer(Strategy)
class ConservativePrePlan:
    """Assumes everyone else is playing DEFAULT_STRAT, then tries to maximize price of
    powerplants without upsetting the aggreggate supply function, thus earning more."""
    # HACK just so hacky
    def set_prices(self, portfolios, hour, plants):
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

@implementer(Strategy)
class CancerComboStrat:
    def __init__(self, amt):
        self.amt = amt
    def set_prices(self, portfolios, hour, plants):
        copy = portfolios.copy()
        prices = []
        for p in copy:
            for plant in p.plants:
                prices.append(Price(plant, p.name, plant.capacity, plant.var_cost))
        prices.sort()
        x = 0
        pt = 0
        demand = Demand(hour)
        for price in prices:
            d = demand.f(x+price.capacity)
            if d < price.value:
                pt = price.value
                break
            x += price.capacity

        for plant in plants:
            if type(self.amt) == list:
                actual_pt = pt-self.amt[hour % 4]
            elif callable(self.amt):
                actual_pt = pt-self.amt(hour)
            else:
                actual_pt = pt-self.amt

            for i, price in enumerate(prices):
                if price.src.name == plant.name and i+1 < len(prices):
                    # print(plant.name)
                    # plant = plants[plants.index(price.src)]
                    if plant.var_cost < actual_pt:
                        # print("hi")
                        plant.price = actual_pt
                        continue
                    next_price = prices[i+1]
                    counter = 1
                    cont_loop = True
                    while i+counter < len(prices)-1 and cont_loop:
                        # print("AAA", prices[i+counter].value)
                        plant.price = max(prices[i+counter].value-0.001, 0)
                        if next_price.src not in plants:
                            cont_loop = False
                            continue                            
                            
                        counter += 1
                        next_price = prices[i+counter]
                    
                    if plant.price is None:
                        plant.price = plant.var_cost                        
                    
            
            # if plant.var_cost < actual_pt:
            #     plant.price = actual_pt
            # else:




@implementer(Strategy)
class RealIntersection:
    def set_prices(self, portfolios, hour, plants):
        copy = portfolios.copy()
        prices = []
        for p in copy:
            for plant in p.plants:
                prices.append(Price(plant, p.name, plant.capacity, plant.var_cost))
        prices.sort()

        x = 0
        pt = 0
        demand = Demand(hour)
        for price in prices:
            d = demand.f(x+price.capacity)
            if d < price.value:
                pt = price.value
                break
            x += price.capacity

        for plant in plants:
            plant.price = max(pt, plant.var_cost)


@implementer(Strategy)
class AggroRealIntersection:
    def __init__(self, mult):
        self.mult = mult

    def set_prices(self, portfolios, hour, plants):
        copy = portfolios.copy()
        prices = []
        for p in copy:
            for plant in p.plants:
                prices.append(Price(plant, p.name, plant.capacity, plant.var_cost))
        prices.sort()

        x = 0
        pt = 0
        demand = Demand(hour)
        for price in prices:
            d = demand.f(x+price.capacity)
            if d < price.value:
                pt = price.value
                break
            x += price.capacity

        for plant in plants:
            plant.price = max(pt*self.mult, plant.var_cost)


@implementer(Strategy)
class UndercutRealIntersection:
    def __init__(self, amt):
        self.amt = amt

    def set_prices(self, portfolios, hour, plants):
        copy = portfolios.copy()
        prices = []
        for p in copy:
            for plant in p.plants:
                prices.append(Price(plant, p.name, plant.capacity, plant.var_cost))
        prices.sort()

        x = 0
        pt = 0
        demand = Demand(hour)
        for price in prices:
            d = demand.f(x+price.capacity)
            if d < price.value:
                pt = price.value
                break
            x += price.capacity
        print("INTERSECTION", pt)

        for plant in plants:
            if type(self.amt) == list:
                plant.price = max(pt-self.amt[hour % 4], plant.var_cost)
            elif callable(self.amt):
                actual_pt = pt-self.amt(hour)
            else:
                plant.price = max(pt-self.amt, plant.var_cost)


    # LATER
# @implementer(Strategy)
# class AggroMarkup:
#     # HACK just so hacky
#     def set_prices(portfolios, hour, plants):
#         copy = portfolios.copy()
#         prices = []
#         for p in copy:
#             DEFAULT_STRAT.set_prices(portfolios, hour, p.plants)
#             for plant in p.plants:
#                 prices.append(Price(plant, p.name, plant.capacity, plant.price))
#         prices.sort()
#         d = Demand(hour)
#         for i, price in enumerate(prices):
#             if price.src in plants and i+1 < len(prices):
#                 plant = plants[plants.index(price.src)]
#                 next_price = prices[i+1]
#                 counter = 1
#                 while i+counter < len(prices)-1:
#                     plant.price = max(prices[i+counter].value-0.001, 0)
#                     if next_price.src not in plants:
#                         break
#                     counter += 1
#                     next_price = prices[i+counter]

DEFAULT_STRAT = SlightProfit # NaiveBreakeven

def get_portfolios():
    portfolios = []
    for f in os.listdir("./portfolios"):
        df = pd.read_csv(f"./portfolios/{f}")
        strat = RealIntersection # Intersection # random.choice([NaiveBreakeven, SlightProfit, SlightlyLessNaiveBreakeven])
        # strat = PessimisticBreakeven
        portfolio = Portfolio(f[:-4], [], strat, 0)
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

def run_sim(portfolios, vv = False, graph = False, start=1):
    for hour in range(start,17):
        for p in portfolios:
            p.set_prices(portfolios, hour)
        prices = aggregate_prices(portfolios)
        steps = get_step_representation(prices)

        print(hour)
        for p in portfolios:
            if p.name == "fossil_light":
                for plant in p.plants:
                    print(plant.name, plant.price)

        demand = Demand(hour)
        for step in steps:
            d = demand.f(step[0]+step[2].capacity)
            if step[1] > d:
                # if step[2].asker == "big_coal":
                #     print(((step[1]-d)/demand.slope))
                step[2].src.used = step[2].capacity + ((step[1]-d)/demand.slope)
                break
            else:
                step[2].src.used = step[2].capacity
                
        if vv:
            print(f"HOUR {hour}")
        for p in portfolios:
            p.calc_gain(eod=(hour % 4 == 0), bod=(hour % 4 == 1))
        if vv:
            for p in reversed(sorted(portfolios)):
                print(p.name, p.money)
            print()
        if graph:
            plot_hour(portfolios, prices, hour)

# def get_net_gain(agg_prices, portfolio, hour):

# debts = {
#     "big_coal": 160e3,
#     "big_gas": 53e3,
#     "bay_views": 210e3,
#     "beachfront": 60e3,
#     "east_bay": 100e3,
#     "old_timers": 650e3,
#     "fossil_light": 832e3,
# }


# debts = {
#     "big_coal": -154927.00,
#     "big_gas": -24376.53,
#     "bay_views": -79707.50,
#     "beachfront": 30531.19,
#     "east_bay": 28203.62,
#     "old_timers": -343771.25,
#     "fossil_light": -435459.35,
# }


debts = {
    "big_coal": -118800.62,
    "big_gas": -2492.36,
    "bay_views": -2600.67,
    "beachfront": 65422.25,
    "east_bay": 59623.80,
    "old_timers": -164724.81,
    "fossil_light": -216156.43,
}




# diff = {
#     "big_coal": -20000.00 + 4750.00 + 10450.00 + 47273.00 + -20000.00,
#     "big_gas": -13642.50 + -2942.50 + -3995.50 + -4481.50 + 72617.50 + -12481.50,
#     "bay_views": 11053.10+17951.40+25976.40+-3351.95+11053.10+18648.05+71257.90+-3351.95,
#     "beachfront": 2128.00+6564.56+28785.00+-15609.50+2047.00+6279.00+83779.50+-18385.77,
#     "east_bay": 1293.50+5337.50+13123.00+-9904.50+1404.00+5782.50+131142.64+-10217.50,
#     "old_timers": 39465.00+45865.00+54190.00+4465.00+45140.00+53040.00+115449.50+8040.00,
#     "fossil_light": 54808.00+60496.00+65585.50+28508.26+60138.05+64865.45+108197.50+28752.00,
# }

def margins(hour):
    if hour == 15:
        return .01 # HEAVY SURGE
    elif hour == 11:
        return .01 # SURGE
    return .01   # REGULAR

for j in range(1):
    portfolios = get_portfolios()
    portfolios[1].strategy = RealIntersection
   #  portfolios[j].strategy = Asshole

    for p in portfolios:
        if p.name == "fossil_light": # us
            p.strategy = CancerComboStrat(margins)
        if p.name == "old_timers": # annli/gigi
            # Pessimistic Scenario
            p.strategy = UndercutRealIntersection(0)
            # p.strategy = AggroRealIntersection(1.16)
            # Optimistic Scenario
            # p.strategy = UndercutRealIntersection(.25)
        if p.name == "big_gas": # hazel/nico/xander
            # Pessimistic Scenario
            # p.strategy = NaiveBreakeven(5)
            p.strategy = UndercutRealIntersection(0)
            # Neutral Scenario
            #p.strategy = AggroRealIntersection(1)
            # Optimistic Scenario
            # p.strategy = AggroRealIntersection(1.10)
        if p.name == "east_bay": # alex/adam/nixie
            # Pessimistic
            p.strategy = UndercutRealIntersection(0)
            # p.strategy = AggroRealIntersection(1.28)
            # Optimistic
            # p.strategy = UndercutRealIntersection(.25)
        if p.name == "big_coal": # anya/laura/nick
            # Pessimistic
            # p.strategy = NaiveBreakeven(10)dis
            # Optimistic
            p.strategy = UndercutRealIntersection(0)
            # p.strategy = AggroRealIntersection(1.20)
        if p.name == "bay_views": # daniel/misha
            # Pessimistic
            # p.strategy = NaiveBreakeven(10)
            # Optimistic
            # p.strategy = AggroRealIntersection(1.18)
            p.strategy = UndercutRealIntersection(0)
        if p.name == "beachfront": # riley/coco
            # Pessimistic
            # p.strategy = NaiveBreakeven(10)
            # Optimistic
            p.strategy = UndercutRealIntersection(0)
            # p.strategy = AggroRealIntersection(1.5)

        p.money = debts[p.name]
        # p.money += diff[p.name]

    # for p in portfolios:
    #     print(p.name, sum([k.capacity for k in p.plants]))
    # portfolios[j].strategy = Asshole
    #portfolios[i].money -= 10000
    # plot_hour(portfolios, prices, 0)

    # print("\n", portfolios[1].name, "\n")
    # if portfolios[j].name == "big_coal":
    #     run_sim(portfolios)#, vv = True , graph=True)
    # else:
    for p in reversed(sorted(portfolios)):
        print(p.name, p.money)
    run_sim(portfolios, vv=True, start=13)#, vv=True,graph=True)
    print()
    for p in reversed(sorted(portfolios)):
        print(p.name, p.money)
    #     p.money = -debts[p.name]

    # run_sim(portfolios, vv = True)
    #     # print(hour)
    #     # plot_hour(portfolios, prices, hour)

    
