def j_to_a(js):
    a = []
    l = len(js)
    for i in range(len(js)):
        a.append((js[(i+1) % l] - js[(i+2) % l]) / js[i])
    return a

print(j_to_a([0.5, 1.0, 1.1]))

print(j_to_a([0.45, 0.5, 0.55]))


# [-0.1111111111111112, 0.20000000000000007, -0.09090909090909088]
# -0.1, 0.2, -0.09

