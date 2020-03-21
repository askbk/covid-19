import numpy as np
import pandas
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit


def find_inflection(time_series, delta):
    """Find the inflection point in a time series"""

    daily_change = np.diff(time_series)

    def safe_ratio(a, b):
        if b == 0:
            if a == 0:
                return 0

            return 2

        return a/b

    ratios = np.array([safe_ratio(daily_change[i], daily_change[i-1]) for i in range(1, len(daily_change))])

    
    # iterate backwards to find where consecutive days of ratio < 1 starts.
    # Require at least 3 days of ratios not greater than 1.
    for day in range(len(ratios) - 1, 1, -1):
        if ratios[day] >= 1:
            if day + 3 < len(ratios) and ratios[day + 1] < 1:
                return day + 1

            break

    raise Exception('No inflection point found')


def find_infection_end(time_series, start_date):
    """Given time series of accumulated infection cases, calculate ending time 
    using inflection point of time series."""

    for day in range(len(time_series)):
        if time_series[day] > 0:
            outbreak_start = day
            break

    try:
        end = find_inflection(time_series[outbreak_start:], 0.05)
        end_date = start_date + datetime.timedelta(days=outbreak_start+2*end)

        return end_date, True
    except:
        return 'No end in sight!', False


def aggregate_by_row_label(data):
    """Combine rows with same label with summation."""

    labels = dict()
    aggregate = np.zeros((1, len(data[0]) - 4))

    for row in data:
        label = row[1]
        row_data = np.array([row[i] for i in range(4, len(row))])
        
        # only aggregate data outside china
        if not label == "China":
            aggregate += row_data

        if label in labels:
            labels[label] += row_data
        else:
            labels[label] = row_data

    return labels, aggregate[0]


def logistic_growth(t, L, k, t_0):
    return L / (1 + np.exp(-k*(t-t_0)))

def logistic_growth_d(t, L, k, t_0):
    return (L * k * np.exp(-k * (t - t_0))) / np.exp(-k * (t - t_0))**2

start_date = datetime.date(2020, 1, 22)

loc = 'COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
covid_data = pandas.read_csv(loc).values
#print(covid_data)

country_data, world_data = aggregate_by_row_label(covid_data)

def print_all():
    found = list()
    for country, data in country_data.items():
        end, has_end = find_infection_end(data, start_date)
        if has_end:
            found.append(f'{country}\t\t{end}')
    
    print('Countries with end dates:')
    for country in found:
        print(country)

    end, has_end = find_infection_end(world_data, start_date)

    print(f'World status: {end}')

    usa_data = country_data['US']
    date_range = mdates.drange(start_date, datetime.date.today(), datetime.timedelta(days=1))
    
    #plt.plot_date(date_range, usa_data)

    #initial guess for curve fit coefficients
    guess = [300000, 0.1, 50]
    # coefficients and curve fit for curve
    xdata = np.arange(0, len(usa_data))
    popt, pcov = curve_fit(logistic_growth, xdata, usa_data, p0=guess)

    
    #xfuture = np.arange(0, 365)

    # Growth
    plt.plot(xdata, logistic_growth(xdata, *popt))
    plt.plot(xdata, usa_data)
        
    # derivatives
    plt.plot(xdata, logistic_growth_d(xdata, *popt))
    plt.plot(xdata[:-1], np.diff(usa_data))
    
    
    # predictions
    #plt.plot(xfuture, logistic_growth(xfuture, *popt))
    #plt.plot(xfuture, logistic_growth_d(xfuture, *popt))

    plt.show()

    #plt.plot_date(date_range, v_fit)
    #plt.show()


print_all()
