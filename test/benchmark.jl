using Nanoflann
using Base.Test

function bench(tree, query_points)
    knn(tree, query_points, 10)
end

N = 100000
L = 10
data = L*rand(Float64, 3, N)
query_points = L*rand(Float64, 3, N)
tree1 = Nanoflann.KDTree(data)
tree2 = NearestNeighbors.KDTree(data)

@time Nanoflann.KDTree(data)
@time NearestNeighbors.KDTree(data)

bench(tree1, query_points[:,1:2])
bench(tree2, query_points[:,1:2])
@time inds1,dist1 = bench(tree1, query_points)
@time inds2,dist2 = bench(tree2, query_points)

for i = 1:N
    @assert vec(inds1[sortperm(dist1[:,i]),i]) == inds2[i][sortperm(dist2[i])]
end

# using Displaz
# inds,_ = knn(tree1, 0.5*L*ones(3), 100)
# Displaz.clf()
# Displaz.plot(data')
# Displaz.hold(true)
# Displaz.plot([0.5,0.5,0.5]', color="b")
# Displaz.plot(data[:,inds]', color="r")

nothing
