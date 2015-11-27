module Nanoflann

export KDTree, knn

using NearestNeighbors

const nanoflann_lib = :nanoflann

type KDTree
    ptr::Ptr{Void}
end

function _free(tree::KDTree)
    ccall((:nanoflann_free_tree, nanoflann_lib), Cint, (Ptr{Void},), tree.ptr)
end

function KDTree(data)
    ptr = ccall((:nanoflann_build_tree, nanoflann_lib),
                Ptr{Void}, (Ptr{Cdouble}, Cint, Csize_t),
                data, size(data, 1), size(data, 2))
    tree = KDTree(ptr)
    finalizer(tree, _free)
    tree
end

function NearestNeighbors.knn(tree::KDTree, point, k)
    inds = Vector{Csize_t}(k)
    dists2 = Vector{Cdouble}(k)
    ccall((:nanoflann_knn, nanoflann_lib),
          Cint, (Ptr{Void}, Ptr{Cdouble}, Cint, Cint, Ptr{Csize_t}, Ptr{Cdouble}),
          tree.ptr, point, length(point), k, inds, dists2) != 0 || error("Error in nanoflann_knn")
    (map(Int, inds)+1, dists2)
end

end # module
