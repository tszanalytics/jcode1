# no real changes made!
using Distributions
using Plots
pyplot(size = (300,300))
# one line version:
# g = rand(Gamma(3,100),1000)
g = Gamma(3,10)
xg = rand(g,1000)
p1 = Plots.plot(xg,color="black")
p2 = Plots.histogram(xg,bins=100,color=[:orange],grid=false)
Plots.plot(p1,p2,layout=2)

savefig("test1.png")

Plots.plot(p2)

ys = Vector[rand(10),rand(20)]
plot(ys,color=[:black :orange])

# distributions usage
mean(g)
std(g)

println("mode:", mode(g), " skewness:", skewness(g),
        " kurtosis:", kurtosis(g), " median:", median(g))

quantile(Normal(), [0.5, 0.95])

# parameters of the theoretical distribution
modes(g)  # array of all modes
median(g)
std(g)
var(g)

# sample parameter estimates
mean(xg)
median(xg)
mode(xg) # should create bins for this
std(xg)

l = -1.0
u = Inf
tn = Truncated(Normal(0, 1), l,u)
xtn = rand(tn,1000)
samplepdf = histogram(xtn,bins=100,color=[:green],grid=false,normed=true)
plot(samplepdf)
z = (1:1100)/200 - 2.0
trupdf = pdf(tn,z)
plot(x=z,y=trupdf)
plot!(samplepdf)

# How to overlay above two plots??
using Gadfly
Gadfly.plot(layer(x=rand(10), y=rand(10), Geom.point),
     layer(x=rand(10), y=rand(10), Geom.line))

# following works
Gadfly.plot(layer(x=rand(10), y=rand(10), Geom.point),
      layer(x=rand(10), y=rand(10), Geom.histogram))

# following works
x = rand(100)
Gadfly.plot(layer(x=x, Geom.density),
            layer(x=x, Geom.histogram))

g = Gamma(3,10)
xg = rand(g,10000)
Gadfly.plot(layer(x=xg, Geom.density),
                        layer(x=xg, Geom.histogram(density=true)))

Theme(default_color=color("lightgray")

g = Gamma(3,5)
xg = rand(g,500000)
Gadfly.plot(layer(x=xg, Geom.density,Theme(default_color=colorant"blue")),
                        layer(x=xg, Geom.histogram(density=true),
                        Theme(default_color=colorant"green")))

mean(xg)
mean(g)
mode(g)
std(g)
std(xg)

Gadfly.plot(layer(x=xg, Geom.density,Theme(default_color=color("blue"))),
layer(x=xg, Geom.histogram(density=true),
Theme(default_color=color("green"))))

p1 = plot(x=xg, Geom.histogram(density=true),
Theme(default_color=color("green")))

p2 = plot(x=xg, Geom.density,Theme(default_color=color("blue")))

Geom.vstack(p1,p2)


xtn2 = rand(tn,1000000)
l1 = Gadfly.layer(x=xtn, Geom.histogram(bincount=100, density = true))
l2 = Gadfly.layer(x=xtn2,Geom.density)
Gadfly.plot(l1,l2)

Gadfly.plot(l1)
Gadfly.plot(l2)

Gadfly.plot()





using DataFrames
xp=DataFrame(x=linspace(0,15,1501),dens=pdf(Distributions.Gamma(3,1),linspace(0,15,1501)))
using PyPlot
PyPlot.plt.hist(x,nbins,normed="True")
PyPlot.plt.plot(xp[:x],xp[:dens])


using Gadfly
Gadfly.plot(layer(x=xdf[:xmin],y=xdf[:dens],Geom.bar,Theme(default_color=color("lightgray"))),
layer(x=xp[:x],y=xp[:dens],Geom.line,Theme(default_color=color("orange"))))




## example of subplots and color choices
plot(Plots.fakedata(100,10),layout=4,palette=[:grays :blues :heat :lightrainbow],bg_inside=[:orange :pink :darkblue :black])


## creating subplots (a frame of plots)
l = @layout([a{0.1h}, b[c, d e]])
plot(randn(100,5),layout=l,t=[:line :histogram :scatter :steppre :bar],leg=false,ticks=nothing,border=false)


## example from Plots website
y = rand(100)
plot(0:10:100,rand(11,4),lab="lines",w=3,palette=:grays,fill=(0,:auto),α=0.6)
scatter!(y,zcolor=abs(y - 0.5),m=(:heat,0.8,stroke(1,:green)),ms=10 * abs(y - 0.5) + 4,lab="grad")
# gui()  # supposed to send plot to a separate window?!


using KernelDensity
kdg = kde(xg)
x = kdg.x
y = kdg.density

Plots.plot(y=y)
# savefig("kde")


@time rand(g,1000000)
