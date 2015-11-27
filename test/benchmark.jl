using Nanoflann
using Base.Test


# FIXME!
push!(Libdl.DL_LOAD_PATH, "../src")

function bench(tree, points)
    N = size(points,2)
    for i in rand(1:N, N)
        inds,_ = knn(tree, points[:,i], 10)
    end
end

N = 100000
L = 10
data = L*rand(Float64, 3, N)
tree1 = Nanoflann.KDTree(data)
tree2 = NearestNeighbors.KDTree(data)

@time Nanoflann.KDTree(data)
@time NearestNeighbors.KDTree(data)

bench(tree1, data)
bench(tree2, data)
@time foo = bench(tree1, data)
@time foo = bench(tree2, data)


# using Displaz
# inds,_ = knn(tree1, 0.5*L*ones(3), 100)
# Displaz.clf()
# Displaz.plot(data')
# Displaz.hold(true)
# Displaz.plot([0.5,0.5,0.5]', color="b")
# Displaz.plot(data[:,inds]', color="r")
