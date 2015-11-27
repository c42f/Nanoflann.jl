#include <nanoflann.hpp>

// Nanoflann wrapper with simple C knn interface
//
// Compile with
// g++ -O3 -shared -fPIC -o nanoflann.so nanoflann.cpp -I /path/to/nanoflann

namespace {

/// Wrapper to present an DxN array in column major format to nanoflann, where
/// D is the dimensionality of the data, and N is the number of points.
template<int Dim = -1>
class ArrayPointWrapper
{
    public:
        ArrayPointWrapper(const double* points, size_t numPoints, int numDims = Dim)
            : m_points(points),
            m_numPoints(numPoints),
            m_numDims(Dim == -1 ? numDims : Dim)
        { }

        /// Return total number of data points
        inline size_t kdtree_get_point_count() const { return m_numPoints; }

        int numDims() const
        {
            // Statically known Dim should be inlined when possible
            return Dim == -1 ? m_numDims : Dim;
        }

        /// Returns the distance squared between p1[0:size-1] and the point with
        /// index idx stored in the underlying matrix
        inline double kdtree_distance(const double *p1, const size_t idx, size_t /*size*/) const
        {
            double d2 = 0;
            for (int i = 0; i < numDims(); ++i)
            {
                double d = p1[i] - m_points[numDims()*idx + i];
                d2 += d*d;
            }
            return d2;
        }

        /// Returns the i'th component of the idx'th point
        inline double kdtree_get_pt(const size_t idx, int i) const
        {
            return m_points[numDims()*idx + i];
        }

        /// Return false to get nanoflann to compute the bounding box itself
        template <class BBOX>
        bool kdtree_get_bbox(BBOX &bb) const { return false; }

    private:
        const double* m_points;
        size_t m_numPoints;
        int m_numDims;
};


/// Wrapper around a nanoflann kdtree and point cloud adaptor
class TreeWrapper
{
    public:
        virtual ~TreeWrapper() {};

        virtual bool knnSearch(const double* point, int dim, int k,
                               size_t* resultIndices, double* resultDistances2) const = 0;
};


template<int N = -1>
class TreeWrapperN : public TreeWrapper
{
    public:
        TreeWrapperN(const double* points, int numPoints, int numDims)
            : m_pointCloud(points, numPoints, numDims),
            m_index(numDims, m_pointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(20 /* max leaf points */))
        {
            m_index.buildIndex();
        }

        virtual bool knnSearch(const double* point, int dim, int k,
                               size_t* resultIndices, double* resultDistances2) const
        {
            if (dim != m_pointCloud.numDims())
                return false;
            m_index.knnSearch(point, k, resultIndices, resultDistances2);
            return true;
        }

    private:
        typedef nanoflann::L2_Simple_Adaptor<double,ArrayPointWrapper<N> > DistanceAdaptor;
        typedef nanoflann::KDTreeSingleIndexAdaptor<DistanceAdaptor, ArrayPointWrapper<N>, N> Tree;

        std::vector<std::pair<size_t, double> > m_matches;
        ArrayPointWrapper<N> m_pointCloud;
        Tree m_index;
};

}


//------------------------------------------------------------------------------
// C interface
extern "C" {

void* nanoflann_build_tree(const double* data, int dim, size_t npoints)
{
#define DIM_CASE(Dim) if (dim == Dim) return new TreeWrapperN<Dim>(data, npoints, dim);
    // Statically dimensioned trees up to some small size
    DIM_CASE(1);
    DIM_CASE(2);
    DIM_CASE(3);
    DIM_CASE(4);
    DIM_CASE(5);
    return new TreeWrapperN<-1>(data, npoints, dim);
}

int nanoflann_knn(const void* tree, const double* point, int dim, int k,
                  size_t* result_inds, double* result_dist2)
{
    const TreeWrapper* wrapper = static_cast<const TreeWrapper*>(tree);
    return wrapper->knnSearch(point, dim, k, result_inds, result_dist2);
}

int nanoflann_free_tree(void* tree)
{
    delete static_cast<TreeWrapper*>(tree);
}

}
