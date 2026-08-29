# Gorev 2 ve 4 - gradient'leri elle hesapladim, kod ayni sonucu veriyor mu?
# YZ50 Hafta 2

from value import Value


print("--- ORNEK 1: L = (a*b + c) * f ---")

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = Value(-2.0)

e = a * b       # -6.0
d = e + c       #  4.0
L = d * f       # -8.0

L.backward()

# elle hesapladigim degerler:
# L=1, f=4, d=-2, c=-2, e=-2, b=-4, a=6
print("L.data =", L.data)
print("a.grad =", a.grad, " (elle: 6.0)")
print("b.grad =", b.grad, " (elle: -4.0)")
print("c.grad =", c.grad, " (elle: -2.0)")
print("e.grad =", e.grad, " (elle: -2.0)")
print("d.grad =", d.grad, " (elle: -2.0)")
print("f.grad =", f.grad, " (elle: 4.0)")


print()
print("--- ORNEK 2: tek neuron ---")

x1 = Value(2.0)
x2 = Value(0.0)
w1 = Value(-3.0)
w2 = Value(1.0)
bias = Value(6.8813735870195432)

n = x1 * w1 + x2 * w2 + bias
o = n.tanh()

o.backward()

# elle hesapladigim degerler:
# n=0.5, x1=-1.5, w1=1.0, x2=0.5, w2=0.0
print("o.data  =", o.data, " (0.7071 olmali)")
print("n.grad  =", n.grad, " (elle: 0.5)")
print("x1.grad =", x1.grad, " (elle: -1.5)")
print("w1.grad =", w1.grad, " (elle: 1.0)")
print("x2.grad =", x2.grad, " (elle: 0.5)")
print("w2.grad =", w2.grad, " (elle: 0.0)  <- x2=0 oldugu icin")


print()
print("--- PAYLASILAN DUGUM TESTI: b = a + a ---")
aa = Value(3.0)
bb = aa + aa
bb.backward()
print("aa.grad =", aa.grad, " (2.0 olmali, cunku '+=' kullandim)")


print()
print("--- GOREV 4: tanh'i parcalara ayirdim, ayni mi? ---")

# tek parca tanh
p1 = Value(2.0)
q1 = Value(-3.0)
r1 = (p1 * q1 + Value(6.8813735870195432)).tanh()
r1.backward()

# parcalanmis tanh (exp, toplama, bolme ile)
p2 = Value(2.0)
q2 = Value(-3.0)
r2 = (p2 * q2 + Value(6.8813735870195432)).tanh2()
r2.backward()

print("tek parca  : cikti =", r1.data, " grad =", p1.grad)
print("parcalanmis: cikti =", r2.data, " grad =", p2.grad)
print("fark =", abs(p1.grad - p2.grad))


print()
print("--- NUMERICAL KONTROL (gecen haftaki yontem) ---")

h = 0.000001


def hesapla(x1_deger):
    xx1 = Value(x1_deger)
    ww1 = Value(-3.0)
    xx2 = Value(0.0)
    ww2 = Value(1.0)
    bb2 = Value(6.8813735870195432)
    return (xx1 * ww1 + xx2 * ww2 + bb2).tanh().data


sonuc1 = hesapla(2.0)
sonuc2 = hesapla(2.0 + h)
numerical = (sonuc2 - sonuc1) / h

print("backward ile x1.grad =", x1.grad)
print("numerical ile x1     =", numerical)
print("fark =", abs(x1.grad - numerical))
