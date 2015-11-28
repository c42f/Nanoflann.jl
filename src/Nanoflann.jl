module Nanoflann

export KDTree, knn

using NearestNeighbors

include("../deps/deps.jl")

type KDTree
    ptr::Ptr{Void}
end

function _free(tree::KDTree)
    ccall((:nanoflann_free_tree, nanoflann_lib), Void, (Ptr{Void},), tree.ptr)
end

function KDTree(data)
    ptr = ccall((:nanoflann_build_tree, nanoflann_lib),
                Ptr{Void}, (Ptr{Cdouble}, Cint, Csize_t),
                data, size(data, 1), size(data, 2))
    tree = KDTree(ptr)
    finalizer(tree, _free)
    tree
end

function NearestNeighbors.knn(tree::KDTree, points, k)
    dim = size(points,1)
    npoints = size(points,2)
    inds   = Array(Csize_t, k, npoints)
    dists2 = Array(Cdouble, k, npoints)
    ccall((:nanoflann_knn, nanoflann_lib),
          Cint, (Ptr{Void}, Ptr{Cdouble}, Csize_t, Csize_t, Cint, Ptr{Csize_t}, Ptr{Cdouble}),
                 tree.ptr,  points,       dim,     npoints, k,    inds,         dists2) == 1 ||
        error("Error in nanoflann_knn")
    (map(Int, inds)+1, dists2)
end

end # module
