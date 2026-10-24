#include <parallelforge/threading/bounded_queue.hpp>
#include <string>

namespace parallelforge::threading {

// Explicit template instantiations for common types to speed up builds and test compilation
template class BoundedQueue<int>;
template class BoundedQueue<std::size_t>;
template class BoundedQueue<float>;
template class BoundedQueue<std::string>;

} // namespace parallelforge::threading
