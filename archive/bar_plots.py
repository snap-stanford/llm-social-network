# libraries
import numpy as np
import matplotlib.pyplot as plt
 
## width of the bars
#barWidth = 0.15
#
## Choose the height of the blue bars (gender)
#bars1 = [0.863, 0.862, 0.952, 0.92]
#
## Choose the height of the orange bars (race/ethnicity)
#bars2 = [0.819, 0.612, 0.912, 0.434]
#
## Choose the height of the olive bars (age)
#bars3 = [0.993, 0.921, 0.972, 0.6]
#
## Choose the height of the purple bars (religion)
#bars4 = [0.924, 0.768, 0.955, 0.704]
#
## Choose the height of the brown bars (political affiliation)
#bars5 = [0.914, 0.340, 0.835, 0.536]
#
## Choose the height of the error bars (bars1)
#yer1 = [0.430, 0.067, 0.104, 0]
#
## Choose the height of the error bars (bars2)
#yer2 = [0.507, 0.072, 0.098, 0]
#
## Choose the height of the error bars (bars3)
#yer3 = [0.205, 0.044, 0.052, 0]
#
## Choose the height of the error bars (bars4)
#yer4 = [0.393, 0.048, 0.066, 0]
#
## Choose the height of the error bars (bars5)
#yer5 = [0.284, 0.093, 0.088, 0]
#
## The x position of bars
#r1 = np.arange(len(bars1))
#r2 = [x + barWidth for x in r1]
#r3 = [x + 2 * barWidth for x in r1]
#r4 = [x + 3 * barWidth for x in r1]
#r5 = [x + 4 * barWidth for x in r1]
#
## Create blue bars
#plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', yerr=yer1, capsize=7, label='gender')
#
## Create orange bars
#plt.bar(r2, bars2, width = barWidth, color = 'orange', edgecolor = 'black', yerr=yer2, capsize=7, label='race/ethnicity')
#
## Create olive bars
#plt.bar(r3, bars3, width = barWidth, color = 'olive', edgecolor = 'black', yerr=yer2, capsize=7, label='age')
#
## Create purple bars
#plt.bar(r4, bars4, width = barWidth, color = 'purple', edgecolor = 'black', yerr=yer2, capsize=7, label='religion')
#
## Create brown bars
#plt.bar(r5, bars5, width = barWidth, color = 'brown', edgecolor = 'black', yerr=yer2, capsize=7, label='political affiliation')
#
## Draw line at 1 (insignificant homophily)
#plt.axhline(y=1, color='black', linestyle='--')
#
## general layout
#plt.xticks([r + barWidth for r in range(len(bars1))], ['all-at-once', 'llm-as-agent', 'one-by-one', 'literature'])
#plt.ylabel('homophily')
#plt.legend()
#
## Show graphic
#plt.show()

# width of the bars
barWidth = 0.15
 
# Choose the height of the blue bars (all-at-once)
bars1 = [0.863, 0.819, 0.993, 0.924, 0.914]
 
# Choose the height of the orange bars (llm-as-agent)
bars2 = [0.862, 0.612, 0.921, 0.768, 0.340]

# Choose the height of the olive bars (one-by-one)
bars3 = [0.952, 0.912, 0.972, 0.955, 0.835]

# Choose the height of the purple bars (literature review)
bars4 = [0.92, 0.434, 0.6, 0.704, 0.536]

# Choose the height of the error bars (bars1)
yer1 = [0.430, 0.507, 0.205, 0.393, 0.284]
 
# Choose the height of the error bars (bars2)
yer2 = [0.067, 0.072, 0.044, 0.048, 0.093]

# Choose the height of the error bars (bars3)
yer3 = [0.104, 0.098, 0.052, 0.066, 0.088]

# Choose the height of the error bars (bars4)
yer4 = [0, 0, 0, 0, 0]
 
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + 2 * barWidth for x in r1]
r4 = [x + 3 * barWidth for x in r1]
 
# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', yerr=yer1, capsize=7, label='all-at-once')
 
# Create orange bars
plt.bar(r2, bars2, width = barWidth, color = 'orange', edgecolor = 'black', yerr=yer2, capsize=7, label='llm-as-agent')

# Create olive bars
plt.bar(r3, bars3, width = barWidth, color = 'olive', edgecolor = 'black', yerr=yer3, capsize=7, label='one-by-one')

# Create purple bars
plt.bar(r4, bars4, width = barWidth, color = 'purple', edgecolor = 'black', yerr=yer4, capsize=7, label='literature')

# Draw line at 1 (insignificant homophily)
plt.axhline(y=1, color='black', linestyle='--')
 
# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], ['gender', 'race/ethnicity', 'age', 'religion', 'political affiliation'])
plt.ylabel('homophily')
plt.legend()
 
# Show graphic
plt.show()
