import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

class Backtester_futures():
    def __init__(self, data, close_column='Close', date_column='Date', low_column='low', high_column='high',
                 long_fee=0,  short_fee=0, hold_asset=True):
        self.data = data.reset_index().copy()
        self.data["account_balance"] = 0
        self.data["asset_balance"] = 0
        self.data["position_balance"] = 0
        self.data["orders"] = 0
        self.data["deposit_withdraw"] = 0
        self.data['pct_1'] = self.data[close_column].pct_change().fillna(1)
        self.data['long_fee'] = long_fee
        self.data['short_fee'] = short_fee
        self.data['trailing_stopl_long'] = 0
        self.data['trailing_stopl_short'] = 0

        self.trade_history = []
        self.enter_price = None

        self.trailing_stopl_long = False
        self.trailing_stopl_short = False

        self.in_stop_loss = False
        self.stop_loss_pct = 0.02
        self.stop_loss_fee = 0.0
        self.stopl_price = None
        self.stopl_balance = None
        self.stopl_asset = None

        self.in_take_profit = False
        self.take_profit_pct = 0.02
        self.take_profit_fee = 0.0
        self.take_profit_price = None
        self.take_profit_balance = None
        self.take_profit_asset = None

        self.date_column = date_column
        self.close_column = close_column
        self.low_column = low_column
        self.high_column = high_column

        self.buy_commission = 0.0005
        self.sell_commission = 0.0005
        self.deposit_fee = 0.0
        self.withdraw_fee = 0.0
        self.exit_trade_commission = 0.0

        self.cur_ind = 0
        self.show_logs = True

        self.hold_asset = hold_asset

        self.use_pct = True

        self.buy_amount = 100
        self.buy_pct = 1
        self.sell_amount = 100
        self.sell_pct = 1

        self.position = False

    def set_param(self, p):
        self.p = p

    def set_stop_loss(self, pct):
        self.stop_loss_pct = pct

    def set_take_profit(self, pct):
        self.take_profit_pct = pct

    def set_trailing_stopl_long(self, series):
        self.data["trailing_stopl_long"] = series.copy()

    def set_trailing_stopl_long(self, series):
        self.data["trailing_stopl_long"] = series.copy()

    def deposit(self, amount):
        self.data.loc[self.cur_ind,
                      'account_balance'] += (1-self.deposit_fee)*amount
        self.data.loc[self.cur_ind, 'asset_balance'] += (
            1-self.deposit_fee)*amount/self.data.loc[self.cur_ind, self.close_column]
        self.loc[self.cur_ind, 'deposit_withdraw'] = amount

    def withdraw(self, amount):
        if self.position:
            print("Can't withdraw while in trade.")
            return
        self.data.loc[self.cur_ind, 'deposit_withdraw'] -= amount
        self.data.loc[self.cur_ind,
                      'account_balance'] -= (1-self.withdraw_fee)*amount
        self.data.loc[self.cur_ind, 'asset_balance'] -= (
            1-self.withdraw_fee)*amount/self.data.loc[self.cur_ind, self.close_column]

    def set_initial_balance(self, amount):
        self.data.loc[0, 'account_balance'] = amount
        self.data.loc[0, 'asset_balance'] = amount / \
            self.data.loc[0, self.close_column]

    def show_balance(self):
        print(self.data.loc[self.cur_ind, 'account_balance'])

    def set_long_amount(self, x=None, x_pct=None):
        if x:
            self.buy_amount = x
        if x_pct:
            self.buy_pct = x_pct

    def set_short_amount(self, x=None, x_pct=None):
        if x:
            self.sell_amount = x
        if x_pct:
            self.sell_pct = x_pct

    def log(self):
        print(
            f'Time: {self.data.loc[self.cur_ind, self.date_column]}, Price: {self.data.loc[self.cur_ind, self.close_column]}, Balance: {self.data.account_balance.iloc[self.cur_ind]}')

    def plot(self, column='account_balance', baseline=True):
        if baseline == True:
            mult = self.data.loc[self.from_index, 'account_balance'] / \
                self.data.loc[self.from_index, self.close_column]
            plt.plot(self.data.loc[self.from_index:self.to_index-1, self.date_column], self.data.loc[self.from_index:self.to_index-1, self.close_column]*mult,
                     label='Buy and hold')
        plt.plot(self.data.loc[self.from_index:self.to_index-1, self.date_column],
                 self.data.loc[self.from_index:self.to_index-1, column], label='strategy')
#         self.data.plot(y=column, x=self.date_col, label='strategy')
        plt.legend()

    def long(self):
        amount = self.data.loc[self.cur_ind, "account_balance"] * self.buy_pct
        if self.data.loc[self.cur_ind, "account_balance"] > 0:
            self.data.loc[self.cur_ind, 'orders'] = amount * \
                (1-self.buy_commission)

            self.data.loc[self.cur_ind,
                          'position_balance'] += amount*(1-self.buy_commission)
            self.data.loc[self.cur_ind,
                          'account_balance'] -= amount*self.buy_commission
            self.data.loc[self.cur_ind, 'asset_balance'] -= (
                1-self.buy_commission)*amount/self.data.loc[self.cur_ind, self.close_column]

            if self.stop_loss_pct:
                self.stopl_balance = (1-self.stop_loss_pct) * \
                    self.data.loc[self.cur_ind, 'account_balance']
                self.stopl_asset = (1-self.stop_loss_pct) * \
                    self.data.loc[self.cur_ind, 'asset_balance']
                pct = self.stop_loss_pct * \
                    self.data.loc[self.cur_ind, 'account_balance'] / \
                    self.data.loc[self.cur_ind, 'position_balance']
                self.stopl_price = self.data.loc[self.cur_ind,
                                                 self.close_column] * (1-pct)
                self.in_stop_loss = True

            self.position = 'long'
            if self.show_logs:
                print(
                    f'Long order ${amount} at price: {self.data.loc[self.cur_ind, self.close_column]}')

        else:
            print('Not enough money in account.')

    def short(self):

        if self.data.loc[self.cur_ind, "account_balance"] > 0:
            amount = self.data.loc[self.cur_ind,
                                   'account_balance'] * self.sell_pct
            self.data.loc[self.cur_ind, 'orders'] = - \
                amount*(1-self.sell_commission)

            self.data.loc[self.cur_ind, 'position_balance'] = - \
                amount*(1-self.sell_commission)
            self.data.loc[self.cur_ind,
                          'account_balance'] -= amount*(self.sell_commission)
            self.data.loc[self.cur_ind, 'asset_balance'] -= (
                1-self.sell_commission)*amount/self.data.loc[self.cur_ind, self.close_column]

            if self.stop_loss_pct:
                self.stopl_balance = (1-self.stop_loss_pct) * \
                    self.data.loc[self.cur_ind, 'account_balance']
                self.stopl_asset = (1-self.stop_loss_pct) * \
                    self.data.loc[self.cur_ind, 'asset_balance']
                pct = self.stop_loss_pct * \
                    self.data.loc[self.cur_ind, 'account_balance'] / \
                    self.data.loc[self.cur_ind, 'position_balance']
                self.stopl_price = self.data.loc[self.cur_ind,
                                                 self.close_column] * (1-pct)
                self.in_stop_loss = True

            self.position = 'short'
            if self.show_logs:
                print(
                    f'Short order ${amount} at price: {self.data.loc[self.cur_ind, self.close_column]}')
        else:
            print('Not enough money in account.')

    def exit_trade(self):
        self.data.loc[self.cur_ind, 'position_balance'] = 0
        self.position = False

    def update_balance(self):
        pct_change = self.data.loc[self.cur_ind, "pct_1"]
        prev_position = self.data.loc[self.cur_ind-1, "position_balance"]

        if self.position:
            max_drawdown_pct = self.data.loc[self.cur_ind, self.low_column] / \
                self.data.loc[self.cur_ind-1, self.close_column] - 1
            max_drawdown = prev_position * max_drawdown_pct

            if self.in_stop_loss:
                #checking for stop loss
                z = False
                if self.position == 'long' and self.data.loc[self.cur_ind, self.low_column] < self.stopl_price:
                    z = True
                elif self.position == 'short' and self.data.loc[self.cur_ind, self.high_column] > self.stopl_price:
                    z = True

                if z:
                    self.data.loc[self.cur_ind,
                                  'account_balance'] = self.stopl_balance
                    self.data.loc[self.cur_ind,
                                  'asset_balance'] = self.stopl_asset
                    self.data.loc[self.cur_ind, 'position_balance'] = 0
                    self.position = False
                    if self.show_logs:
                        print(
                            f'You have been stopped out by your stop loss at ${self.stopl_price}, you have ${self.stopl_balance} left in your account.')
                    return

            elif self.data.loc[self.cur_ind-1, 'account_balance'] + max_drawdown <= 0:
                #checking liquidation
                self.data.loc[self.cur_ind, 'position_balance'] = 0
                self.data.loc[self.cur_ind, 'account_balance'] = 0
                self.data.loc[self.cur_ind, 'asset_balance'] = 0
                print('You have been liquidated!')
                self.position = False
                return True

            change = prev_position * pct_change
            trading_fee = self.data.loc[self.cur_ind,
                                        "long_fee"] if prev_position > 0 else self.data.loc[self.cur_ind, "short_fee"]

            self.data.loc[self.cur_ind,
                          'position_balance'] -= trading_fee * prev_position
            self.data.loc[self.cur_ind,
                          'account_balance'] -= trading_fee * prev_position
            self.data.loc[self.cur_ind, 'asset_balance'] -= trading_fee * \
                prev_position / self.data.loc[self.cur_ind, self.close_column]

            if self.hold_asset != True:
                pct_change = 1
            else:
                pct_change += 1

            self.data.loc[self.cur_ind,
                          'position_balance'] = self.data.loc[self.cur_ind-1, 'position_balance']
            self.data.loc[self.cur_ind, 'account_balance'] = self.data.loc[self.cur_ind-1, 'account_balance']\
                * pct_change + change
            self.data.loc[self.cur_ind, 'asset_balance'] = self.data.loc[self.cur_ind-1, 'asset_balance']\
                * pct_change + (change/self.data.loc[self.cur_ind, self.close_column])
        else:
            if self.hold_asset != True:
                pct_change = 1
            else:
                pct_change += 1

            self.data.loc[self.cur_ind,
                          'position_balance'] = self.data.loc[self.cur_ind-1, 'position_balance']
            self.data.loc[self.cur_ind, 'account_balance'] = self.data.loc[self.cur_ind -
                                                                           1, 'account_balance'] * pct_change
            self.data.loc[self.cur_ind, 'asset_balance'] = self.data.loc[self.cur_ind -
                                                                         1, 'asset_balance'] * pct_change

        return False

    def run(self, from_date=None, to_date=None, plot=True, show_logs=False):
        self.show_logs = show_logs
        if from_date == None:
            f = 0
        else:
            f = self.data.loc[self.data[self.date_column]
                              == from_date].index[0]
            amount = self.data.loc[0, 'account_balance']
            self.data.loc[f, 'account_balance'] = amount
            self.data.loc[f, 'asset_balance'] = amount / \
                self.data.loc[0, self.close_column]

        if to_date == None:
            t = len(self.data)
        else:
            t = self.data.loc[self.data[self.date_column] == to_date].index[0]

        self.from_index = f
        self.to_index = t

        for i in range(f+1, t):
            self.cur_ind = i
            if self.update_balance():
                break
            self.next_cycle()

            if self.show_logs:
                self.log()

        if plot:
            self.plot()

    def set_ind(self, param):
        if param == None:
            return
        for indicator in self.indicator_functions:
            p = param[indicator.__name__]
            indicator(self.data, **p)

        for ema_ in self.ema_periods:
            name = 'ema_' + str(ema_)
            self.data[name] = ema(self.data, period=ema_,
                                  column=self.close_column, inplace=False)

    def grid_strategy(self, strategy, params):
        pass

    def return_history(self, columns=['account_balance']):
        mult = self.data.loc[self.from_index, 'account_balance'] / \
            self.data.loc[self.from_index, self.close_column]
        columns = [self.date_column, self.close_column] + columns
        z = self.data.loc[self.from_index:self.to_index, columns].copy()
        z[self.close_column] *= mult
        return z

    def print_balance(self):
        print(self.data.loc[self.cur_ind, 'account_balance'])