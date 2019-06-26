import matplotlib.pyplot as plt

T = [0, 8, 6, 8, 0, 0, 0, 6, 4, 0] #[4, 6, 0, 0, 4, 4, 4, 0, 8, 0, 0, 0, 0, 4, 6, 7, 0, 0]
T.reverse()
S = list(range(0, len(T)))


plt.plot(S, T, 'k-')
plt.plot(S, T, 'k')
#plt.axis([0, 6, 0, 20])
plt.show()