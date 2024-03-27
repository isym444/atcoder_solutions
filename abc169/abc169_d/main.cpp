#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <queue>
#include <array>
#include <climits>
#include <cmath>
#include <set>
#include <map>
#include <bitset>
#include <deque>
#include <numeric>
#include <assert.h>
#include <unordered_map>
#include <type_traits> // For std::is_floating_point
#include <cmath> // For std::ceil

using namespace std;

#define REP(i, n) for (int i = 0; (i) < (int)(n); ++ (i))
#define REP3(i, m, n) for (int i = (m); (i) < (int)(n); ++ (i))
#define REP_R(i, n) for (int i = (int)(n) - 1; (i) >= 0; -- (i))
#define REP3R(i, m, n) for (int i = (int)(n) - 1; (i) >= (int)(m); -- (i))
#define ALL(x) ::std::begin(x), ::std::end(x)
#define ll long long
#define sz(x) (int)(x).size()
#define fo(from_0_to_non_incl_to) for(int i=0;i<from_0_to_non_incl_to;i++)
//h CAREFUL if you put an expression as an argument it will give bugs, better assign expression to variable then put that in the foi() as argument
#define foi(from,non_incl_to) for(int i=from;i<(non_incl_to);i++)
#define foii(non_incl_to) for(int i=0;i<(non_incl_to);i++)
#define foj(from,non_incl_to) for(int j=from;j<(non_incl_to);j++)
#define fojj(non_incl_to) for(int j=0;j<(non_incl_to);j++)
#define fok(from,non_incl_to) for(int k=from;k<(non_incl_to);k++)
#define fokk(non_incl_to) for(int k=0;k<(non_incl_to);k++)
#define fa(x, dataStructure) for(auto x : dataStructure)
#define fx(dataStructure) for(auto x : dataStructure)
#define wasd(x) foi(-1,2) foj(-1,2) if(abs(i)+abs(j)==1){x};
#define qweasdzxc(x) foi(-1,2) foj(-1,2) if(abs(i)+abs(j)==1){x};
#define isvalid(x_plus_i,max_boundary_n,y_plus_j,max_boundary_m) (0<=x_plus_i and x_plus_i<max_boundary_n and 0<=y_plus_j and y_plus_j<max_boundary_m)
//#define gcd __gcd
#define mp make_pair
//Makes % get floor remainder (towards -INF) and make it always positive
#define MOD(x,y) (x%y+y)%y
// #define print(p) cout<<p<<endl
#define fi first
#define sec second
#define prmap(m) {for(auto i: m) cout<<(i.fi)<<i.sec<<endl}
#define pra(a) {for(auto i: a) cout<<i<<endl;}
#define prm(a) {for(auto i: a) pra(i) cout<<endl;}
//#define itobin(x) bitset<32> bin(x)
#define itobin(intToConvertTo32BitBinaryNum) std::bitset<32>(intToConvertTo32BitBinaryNum)
#define bintoi(binaryNum32BitToConvertToInt) binaryNum32BitToConvertToInt.to_ulong()
#define binstoi(binaryStringToConvertToInt) stoi(binaryStringToConvertToInt, nullptr, 2)
#define vecsum(vectorName) accumulate((vectorName).begin(), (vectorName).end(), 0)
#define setbits(decimalnumber) __builtin_popcount(decimalnumber)
#define stringSplice(str, i, j) (str).erase(i, j) //j is the length of string to erase starting from index i
#define string_pop_back(str) (str).pop_back()
#define substring(str, i, j) (str).substr(i, j) //j is the length of substring from i
#define rng(a) a.begin(),a.end()
#define all(a) a.begin(),a.end()

typedef pair<ll, ll> pl;
typedef vector<long long> vll;
typedef std::vector<std::vector<long long>> vvll;

#define pb push_back

ll INF=1e18;


/*/---------------------------IO(Debugging)----------------------/*/
template<class T> istream& operator >> (istream &is, vector<T>& V) {
    for(auto &e : V)
        is >> e;
    return is;
}

template<typename CharT, typename Traits, typename T>
ostream& _containerprint(std::basic_ostream<CharT, Traits> &out, T const &val) {
    return (out << val << " ");
}
template<typename CharT, typename Traits, typename T1, typename T2>
ostream& _containerprint(std::basic_ostream<CharT, Traits> &out, pair<T1, T2> const &val) {
    return (out << "(" << val.first << "," << val.second << ") ");
}
template<typename CharT, typename Traits, template<typename, typename...> class TT, typename... Args>
ostream& operator << (std::basic_ostream<CharT, Traits> &out, TT<Args...> const &cont) {
    out << "[ ";
    for(auto&& elem : cont) _containerprint(out, elem);
    return (out << "]");
}
template<class L, class R> ostream& operator << (ostream& out, pair<L, R> const &val){
    return (out << "(" << val.first << "," << val.second << ") ");
}
template<class P, class Q = vector<P>, class R = less<P> > ostream& operator << (ostream& out, priority_queue<P, Q, R> const& M){
    static priority_queue<P, Q, R> U;
    U = M;
    out << "{ ";
    while(!U.empty())
        out << U.top() << " ", U.pop();
    return (out << "}");
}
template<class P> ostream& operator << (ostream& out, queue<P> const& M){
    static queue<P> U;
    U = M;
    out << "{ ";
    while(!U.empty())
        out << U.front() << " ", U.pop();
    return (out << "}");
}
template<typename CharT, typename Traits>
ostream& operator << (std::basic_ostream<CharT, Traits> &out, vector<vector<ll>> const &matrix) {
    for (auto &row : matrix) {
        out << row << "\n";
    }
    return out;
}

void setIO(string name = "")
{ // name is nonempty for USACO file I/O
    ios_base::sync_with_stdio(0);
    cin.tie(0); // see Fast Input & Output
    // alternatively, cin.tie(0)->sync_with_stdio(0);
    if (sz(name))
    {
    freopen((name + ".in").c_str(), "r", stdin); // see Input & Output
    freopen((name + ".out").c_str(), "w", stdout);
    }
}

const int MAXA = 5000006;
bool prime[MAXA];

void gen_primes() {
    fill(prime, prime+MAXA, true);
    prime[0] = 0;
    // prime[1] = 1 for our purposes
    for (int i = 4; i < MAXA; i += 2) {
        prime[i] = 0;
    }

    for (int p = 3; p*p < MAXA; p += 2) {
        if (!prime[p]) continue;
        for (int x = p*p; x < MAXA; x += p) {
            prime[x] = 0;
        }
    }
}

const int MAXN = 2 * 100000 + 10; // Adjust the size as per the problem constraints
//smallest prime factor
std::vector<int> spf(MAXN);

//Sieve of Eratosthenes to precompute the smallest prime factor of each number
void spfsieve() {
    //initializes smallest prime factor for 1 as 1
    spf[1] = 1;
    //initially treats every number as a prime and assigns the smallest prime factor as itself
    for (int i = 2; i < MAXN; ++i) {
        spf[i] = i;
    }
    //all even numbers' smallest prime factor is 2. This sets this for all even numbers
    for (int i = 4; i < MAXN; i += 2) {
        spf[i] = 2;
    }
    //calculates upper limit to which we need to check for primes (optimization)
    //because factors come in pairs e.g. N = 36, its factors are (1, 36), (2, 18), (3, 12), (4, 9), and (6, 6)
    //so after sqrt(N) the pairs will switch and and won't find any further new factors
    //Factors of a number come in pairs, with one factor being less than or equal to the square root of the number,
    //and the other being greater than or equal to the square root.
    int limit = std::ceil(std::sqrt(MAXN));
    for (int i = 3; i < limit; i+=2) {
        //checks if i is still marked as it's own smallest prime factor
        //indicating that it is still prime 
        //because we have iterated through all smaller numbers beforehand so if not a multiple of any of those
        //then by definition it must be prime
        if (spf[i] == i) {
            //start from i*i as spf for smaller numbers than this would be marked by smaller numbers than i (if not prime)
            for (int j = i * i; j < MAXN; j += i) {
                //if smallest prime factor of j is still set as itself i.e. spf not yet been set
                //then set spf as i
                if (spf[j] == j) {
                    spf[j] = i;
                }
            }
        }
    }
}

//Function to return the prime factorization of a number (is unique for every number)
//make sure to call spfsieve() before calling this function so spf values are prepopulated
std::unordered_map<int, int> fact(int x) {
    std::unordered_map<int, int> pfactors;
    while (x != 1) {
        //if the smallest prime factor of x has not yet been added to pfactors
        //  then set p^1
        //if the smallest prime factor of x has already been added to pfactors
        //  then set p^(prev exponent+1)
        if (pfactors.find(spf[x]) == pfactors.end()) {
            pfactors[spf[x]] = 1;
        } else {
            pfactors[spf[x]] += 1;
        }
        //while x!=1, divide x by its smallest prime factor
        x = x / spf[x];
    }
    return pfactors;
}

ll mod=1e9+7;
//ll mod=1000;
//modular exponentiation: calculates a^b mod c where a^b is a crazy big number and would usually overflow. Change mod above as needed
ll mpow(ll base, ll exp)
{
    base %= mod;
    ll result = 1;
    while (exp > 0)
    {
        if (exp & 1)
            result = ((ll)result * base) % mod;
        base = ((ll)base * base) % mod;
        exp >>= 1;
    }
    return result;
}

//use if possible as faster than nCx
long long nC2(int n) {
    return static_cast<long long>(n) * (n - 1) / 2;
}

//order matters so larger than nCx
long long nPx(int n, int x) {
    if (x > n) return 0;
    
    long long result = 1;
    for (int i = n; i > n - x; --i) {
        result *= i;
    }
    return result;
}

//order doesn't matter so smaller than nPx
long long nCx(int n, int x) {
    if (x > n) return 0;
    if (x * 2 > n) x = n - x; // Take advantage of symmetry, nCx == nC(n-x)
    if (x == 0) return 1;

    long long result = 1;
    for (int i = 1; i <= x; ++i) {
        result *= n - (x - i);
        result /= i;
    }
    return result;
}

vector<ll> parent; // To store parent information
//visited nodes
vector<bool> vis;
//bool vis[61][61][61][61]={0};
map<ll,ll> depth;

//initialize graph as adjacency list
vector<vector<ll> > g;
//initialize weighted graph as adjacency list
vector<vector<pair<ll,ll>>> wg;
//for building the adjacency list by adding edges info
ll totalEdges = 0;
void edge(ll originNode, ll destNode)
{
    g[originNode].pb(destNode);
    totalEdges++;
 
    // for undirected graph e.g. tree, add this line:
    // g[destNode].pb(originNode);
}

void edge(ll originNode, ll destNode, ll weight){
    wg[originNode].emplace_back(destNode, weight);
    totalEdges++;
    // For an undirected graph e.g., tree, add this line:
    // g[destNode].emplace_back(originNode, weight);
}

//returns vector where each index is the shortest distance between the start node and node i
vector<ll> dijkstra(ll start) {
    vector<ll> dist(wg.size(), INF);  // Distance from start to each node
    //arguments: 1) type of elements pq will store 2) underlying container to be used by pq 
    //3) comparison function to specify order of elements in pq (default is less with largest element at top i.e. max-heap vs min-heap below)
    priority_queue<pair<ll, ll>, vector<pair<ll, ll>>, greater<pair<ll, ll>>> pq;
    dist[start] = 0;
    pq.push({0, start});  // {distance, node}

    while (!pq.empty()) {
        //cerr << "pq" << pq << endl;
        ll currentDist = pq.top().first;
        ll currentNode = pq.top().second;
        pq.pop();

        // If the distance in priority queue is larger, we have already found a better path
        if (currentDist > dist[currentNode]) {
            continue;
        }
        /* Optimization to try if TLEing instead of if statement above
        if (cdist != dist[node]) { continue; }*/
    
        for (auto &neighbor : wg[currentNode]) {
            ll nextNode = neighbor.first;
            ll weight = neighbor.second;
            ll newDist = currentDist + weight;

            if (newDist < dist[nextNode]) {
                dist[nextNode] = newDist;
                pq.push({newDist, nextNode});
            }
        }
    }

    return dist;
}

//Bellman Ford Graph: L node, R node, weight of edge between L&R
vector<tuple<int, int, ll>> bfg;

vector<ll> bellmanDistances;
// Function to run Bellman-Ford algorithm given start node and # vertices, saves min distances in bellmanDistances vector
bool bellmanFord(ll start, ll V) {
    vector<ll> distance(V, INF);
    bellmanDistances.resize(V, INF);
    bellmanDistances[start] = 0;
    ll a,b,w;
    for (ll i = 1; i <= V - 1; i++) {
        for (const auto& e : bfg) {
            tie(a, b, w) = e;
            bellmanDistances[b]=min(bellmanDistances[b],bellmanDistances[a]+w);
            /* if (bellmanDistances[a] + w < bellmanDistances[b]) {
                bellmanDistances[b] = bellmanDistances[a] + w;
            } */
        }
    }

    // Check for negative-weight cycles (negative sum of edges in a cycle)
    for (const auto& e : bfg) {
        tie(a, b, w) = e;
        if (bellmanDistances[a] + w < bellmanDistances[b]) {
            cout << "Graph contains negative weight cycle" << endl;
            return false;
        }
    }

/*     // Print the distances
    for (int i = 0; i < V; i++)
        cout << "Distance from " << start << " to " << i << " is " << (distance[i] == INF ? "INF" : to_string(distance[i])) << endl;
 */
    return true;
}


//traverse a graph using bfs from the specified start node to all other nodes, in the process printing the order the nodes are visited in
void bfs(ll start)
{
    queue<ll> q;
 
    q.push(start);
    vis[start] = true;
    depth[start] = 1; // Depth of starting node is 1
    //If want first time something happens/node reached then break when that happens
    while (!q.empty()) {
        ll f = q.front();
        q.pop();
 
        cerr << f << " ";
        
        // Enqueue all adjacent of f and mark them visited 
        ll counter = 0;
        for (auto i = g[f].begin(); i != g[f].end(); i++) {
            if (!vis[*i]) {
                counter++;
                q.push(*i);
                vis[*i] = true;
                depth[*i] = depth[f] + 1; // Set the depth of the neighboring node
            }
        }
        /* if(counter==0){
            cerr << "depths to leafs: " << depth[f] << endl;
        } */
    }
}

//bfs function returning vector with the shortest paths from start node to every other node
vector<ll> bfs_shortest_paths(ll start) {
    vector<long long> distances(g.size()+1, -1);
    queue<int> q;

    distances[start] = 0;
    q.push(start);

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        for (int neighbor : g[node]) {
            if (distances[neighbor] == -1) {
                distances[neighbor] = distances[node] + 1;
                q.push(neighbor);
            }
        }
    }
    return distances;
}

//return a vector containing bfs path from start to end nodes specified
vector<ll> bfs(ll start, ll end) {
    queue<ll> q;
    q.push(start);
    vis[start] = true;
    parent[start] = -1; // Start node has no parent

    while (!q.empty()) {
        ll f = q.front();
        q.pop();

        if (f == end) break; // Stop if we reach the end node

        for (auto i = g[f].begin(); i != g[f].end(); i++) {
            if (!vis[*i]) {
                q.push(*i);
                vis[*i] = true;
                parent[*i] = f; // Set parent
            }
        }
    }

    vector<ll> path;
    for (ll i = end; i != -1; i = parent[i]) {
        path.push_back(i);
    }
    reverse(path.begin(), path.end()); // Reverse to get the correct order
    return path;
}

//dfs traversal from start node, in process keeping track of max depth & keeping track of each node's depth & printing order of traversal
ll maxDepth = 0;
void dfs(ll startNode, ll startDepth){
    vis[startNode] = true;
    depth[startNode]=startDepth;
    maxDepth=max(maxDepth, startDepth);
    cerr << startNode << " ";
    for(auto adjNode : g[startNode]){
        if(!vis[adjNode]) dfs(adjNode, startDepth+1);
    }
}

map<ll,ll>subtreeSizes; //Map to store subtree sizes for each child of the start node
ll dfsSubtreesHelper(ll startNode){
    vis[startNode] = true;
    ll subtreeSize = 1;
    //cerr << startNode << " ";
    for(auto adjNode : g[startNode]){
        if(!vis[adjNode]){
            subtreeSize+=dfsSubtreesHelper(adjNode);
        }
    }
    return subtreeSize;
}
//main function to call to populate subtreeSizes
ll minSubtreeSize = 3*pow(10,5)+1; //Adjust this to the max given boundary of the problem
void dfsSubtrees(ll startNode){
    vis[startNode] = true;
    for(auto adjNode : g[startNode]){
        subtreeSizes[adjNode]=dfsSubtreesHelper(adjNode); //+1 if want to include startNode in size of subtrees
        minSubtreeSize=min(minSubtreeSize,subtreeSizes[adjNode]);
    }
}


//disjoint set union/union find
//consider using coordinate compression!
struct dsu {
  public:
    dsu() : _n(0) {}
    //constructor for dsu. Initialize as "dsu name_of_object(n);"
    //creates an undirected graph with n vertices and 0 edges
    //N.b. if initializing for HxW grid then = HxW
    explicit dsu(int n) : _n(n), parent_or_size(n, -1) {}

    //returns representative of component if a&b already in component or else joins them into a new component and selects one as representative
    //don't forget nodes are 0 indexed!!!!!!!!!!!! so if edge in problem connects node 1&2 where nodes are 1 indexed in problem, do --
    int merge(int a, int b) {
        assert(0 <= a && a < _n);
        assert(0 <= b && b < _n);
        int x = leader(a), y = leader(b);
        if (x == y) return x;
        if (-parent_or_size[x] < -parent_or_size[y]) std::swap(x, y);
        parent_or_size[x] += parent_or_size[y];
        parent_or_size[y] = x;
        return x;
    }

    //returns whether a&b in same component
    //N.b. if for HxW grid then have to convert ij 2d coordinate to 1d representation:
    //v = i*W+j
    bool same(int a, int b) {
        assert(0 <= a && a < _n);
        assert(0 <= b && b < _n);
        return leader(a) == leader(b);
    }

    //returns representative of connected component in which a resides
    int leader(int a) {
        assert(0 <= a && a < _n);
        if (parent_or_size[a] < 0) return a;
        return parent_or_size[a] = leader(parent_or_size[a]);
    }

    //returns size of connected component in which a resides
    int size(int a) {
        assert(0 <= a && a < _n);
        return -parent_or_size[leader(a)];
    }

    //returns a list of the nodes of each connected component
    std::vector<std::vector<int>> groups() {
        std::vector<int> leader_buf(_n), group_size(_n);
        for (int i = 0; i < _n; i++) {
            leader_buf[i] = leader(i);
            group_size[leader_buf[i]]++;
        }
        std::vector<std::vector<int>> result(_n);
        for (int i = 0; i < _n; i++) {
            result[i].reserve(group_size[i]);
        }
        for (int i = 0; i < _n; i++) {
            result[leader_buf[i]].push_back(i);
        }
        result.erase(
            std::remove_if(result.begin(), result.end(),[&](const std::vector<int>& v) { return v.empty(); }),result.end());
        return result;
    }

  private:
    int _n;
    // root node: -1 * component size
    // otherwise: parent
    std::vector<int> parent_or_size;
};


template <typename T, bool merge_adjacent_segment = true>
//https://atcoder.jp/contests/abc256/submissions/32542720
//initialize as RangSet<int> rs;
struct RangeSet : public std::map<T, T> {
    public:
        RangeSet() : _size(0) {}

        // returns the number of intergers in this set (not the number of ranges). O(1)
        T size() const { return number_of_elements(); }
        // returns the number of intergers in this set (not the number of ranges). O(1)
        T number_of_elements() const { return _size; }
        // returns the number of ranges in this set (not the number of integers). O(1)
        int number_of_ranges() const { return std::map<T, T>::size(); }

        // returns whether the given integer is in this set or not. O(log N)
        bool contains(T x) const {
            auto it = this->upper_bound(x);
            return it != this->begin() and x <= std::prev(it)->second;
        }

        /**
         * returns the iterator pointing to the range [l, r] in this set s.t. l <= x <= r.
         * if such a range does not exist, returns `end()`.
         * O(log N)
         */
        auto find_range(T x) const {
            auto it = this->upper_bound(x);
            return it != this->begin() and x <= (--it)->second ? it : this->end();
        }

        // returns whether `x` and `y` is in this set and in the same range. O(log N)
        bool in_the_same_range(T x, T y) const {
            auto it = get_containing_range(x);
            return it != this->end() and it->first <= y and y <= it->second;
        }

        // inserts the range [x, x] and returns the number of integers inserted to this set. O(log N)
        T insert(T x) {
            return insert(x, x);
        }
        
        // inserts the range [l, r] and returns the number of integers inserted to this set. amortized O(log N)
        T insert(T l, T r) {
            if (l > r) return 0;
            auto it = this->upper_bound(l);
            if (it != this->begin() and is_mergeable(std::prev(it)->second, l)) {
                it = std::prev(it);
                l = std::min(l, it->first);
            }
            T inserted = 0;
            for (; it != this->end() and is_mergeable(r, it->first); it = std::map<T, T>::erase(it)) {
                auto [cl, cr] = *it; 
                r = std::max(r, cr);
                inserted -= cr - cl + 1;
            }
            inserted += r - l + 1;
            (*this)[l] = r;
            _size += inserted;
            return inserted;
        }

        // erases the range [x, x] and returns the number of intergers erased from this set. O(log N)
        T erase(T x) {
            return erase(x, x);
        }

        // erases the range [l, r] and returns the number of intergers erased from this set. amortized O(log N)
        T erase(T l, T r) {
            if (l > r) return 0;
            T tl = l, tr = r;
            auto it = this->upper_bound(l);
            if (it != this->begin() and l <= std::prev(it)->second) {
                it = std::prev(it);
                tl = it->first;
            }
            T erased = 0;
            for (; it != this->end() and it->first <= r; it = std::map<T, T>::erase(it)) {
                auto [cl, cr] = *it;
                tr = cr;
                erased += cr - cl + 1;
            }
            if (tl < l) {
                (*this)[tl] = l - 1;
                erased -= l - tl;
            }
            if (r < tr) {
                (*this)[r + 1] = tr;
                erased -= tr - r;
            }
            _size -= erased;
            return erased;
        }

        // returns minimum integer x s.t. x >= lower and x is NOT in this set
        T minimum_excluded(T lower = 0) const {
            static_assert(merge_adjacent_segment);
            auto it = find_range(lower);
            return it == this->end() ? lower : it->second + 1;
        }

        // returns maximum integer x s.t. x <= upper and x is NOT in this set
        T maximum_excluded(T upper) const {
            static_assert(merge_adjacent_segment);
            auto it = find_range(upper);
            return it == this->end() ? upper : it->first - 1;
        }

    private:
        T _size;

        bool is_mergeable(T cur_r, T next_l) {
            return next_l <= cur_r + merge_adjacent_segment;
        }
};

string itobins(int n) {
    if (n == 0) return "0";

    string binary = "";
    while (n > 0) {
        binary += (n % 2) ? '1' : '0';
        n /= 2;
    }

    reverse(binary.begin(), binary.end()); // Reverse to get the correct order
    return binary;
}

string dtobx(int decimalNumber, int base) {
    if (base < 2 || base > 36) {
        return "Invalid base";
    }

    string result = "";
    while (decimalNumber > 0) {
        int remainder = decimalNumber % base;

        // Convert remainder to corresponding character
        if (remainder >= 10) {
            result += 'A' + (remainder - 10);
        } else {
            result += '0' + remainder;
        }

        decimalNumber /= base;
    }

    // Reverse the string as the result is calculated in reverse order
    reverse(result.begin(), result.end());

    return result.empty() ? "0" : result;
}

template<typename T, typename U>
auto ceildiv(T n, U d) -> decltype(n / d + 0) {
    static_assert(std::is_arithmetic<T>::value && std::is_arithmetic<U>::value, "ceildiv requires arithmetic types");

    if constexpr (std::is_floating_point<T>::value || std::is_floating_point<U>::value) {
        // Handle case where either n or d is a floating-point number
        return static_cast<decltype(n / d + 0)>(std::ceil(n / static_cast<double>(d)));
    } else {
        // Handle case where both n and d are integers
        return (n + d - 1) / d;
    }
}

/* ll ceildiv(ll n, ll d){
return((n+d-1)/d);
}
*/
/* ll floordiv(ll n, ll d){
ll x = (n%d+d)%d;
return ((n-x)/d);
}
 */
template<typename T, typename U>
auto floordiv(T n, U d) -> decltype(n / d + 0) {
    static_assert(std::is_arithmetic<T>::value && std::is_arithmetic<U>::value, "floordiv requires arithmetic types");

    if constexpr (std::is_floating_point<T>::value || std::is_floating_point<U>::value) {
        // Handle case where either n or d is a floating-point number
        // Perform the division as floating-point operation and use std::floor to round down
        return static_cast<decltype(n / d + 0)>(std::floor(n / static_cast<double>(d)));
    } else {
        // Handle case where both n and d are integers
        // Original logic for floor division with integers
        T x = (n % d + d) % d;
        return (n - x) / d;
    }
}

ll midpoint(ll L, ll R){
    return (L+(R-L)/2);
}

ll lcm(ll a, ll b) {
    return std::abs(a * b) / std::gcd(a, b);
}


int stringToBinary(const std::string& s, char charAsOne) {
    int x = 0;
    for (int j = 0; j < s.length(); j++) {
        x = 2 * x + (s[j] == charAsOne);
    }
    return x;
}

//returns index of first element greater than or equal to target
ll findGreaterEqual(vector<ll> sortedVector, ll target){
    auto it = lower_bound(sortedVector.begin(), sortedVector.end(), target);
    return it-sortedVector.begin();
}

//returns index of first element less than or equal to target
//if all elements are greater than target returns -1
//if all elements are smaller than target, returns last element
//h N.B. gives 1-based index
ll findLessEqual(vector<ll> sortedVector, ll target){
    auto it = upper_bound(sortedVector.begin(), sortedVector.end(), target);
    if(it != sortedVector.begin()){
        --it;
        if(*it<=target){
            return it-sortedVector.begin()+1;
        }
    }
    else{
        return -1;
    }
}

struct loc
{
    ll x;
    ll y;
    char dir;
    loc(ll x, ll y, char c) : x(x), y(y), dir(c) {}
    //loc::x to access or modify x
    //initialize using loc locobj(1,2,'r')
    //if don't want to use constructor, can initialize using loc locobj = {1, 2, 'n'};
};

/* sorting vector<loc> locvector by y first then x
std::sort(locations.begin(), locations.end(), [](const loc &a, const loc &b) {
        if (a.y == b.y) {
            return a.x < b.x; // Sort by x if y is the same
        }
        return a.y < b.y; // Otherwise, sort by y
    });
 */

vector<ll> genAlphabetPlaceholder(){
    vector<ll> f(26);
    iota(f.begin(), f.end(), 0);
    return f;
}

vector<char> genAlphabet(){
    vector<char> alphabet(26);
    iota(alphabet.begin(), alphabet.end(), 'a');
    return alphabet;
}

const ll LCSN = 1000 + 20;
//dp holds the longest common subsequence (LCS) for the 2 substrings to index i & j
//e.g. for s = "hleloworld", t = "thequickbrown"
//i=3 j=4 i.e. dp[3][4]=2 as s[0toi]="hle" & t[0toj]="theq" -> LCS is "he"
ll dp[LCSN][LCSN];
//outputs the value of the longest common subsequence between 2 strings s & t
ll lcs(string s, string t){
    for(ll i = 0; i <= s.size(); i++){
        for(ll j = 0; j <= t.size(); j++){
            //if either s or t is empty then LCS = 0
            if(!i || !j) dp[i][j] = 0;
            //else if cur letters being compared i.e. s[i] or t[i] are the same
            //then dp[i][j] is 1 more than dp[i-1][j-1]
            else if (s[i-1] == t[j-1]) dp[i][j]=dp[i-1][j-1]+1;
            //else if cur letters being compared i.e. s[i] or t[i] are not the same
            //then dp[i][j] is max of dp[i-1][j] (comparing cur longest t to s-1)
            //and dp[i][j-1] (comparing cur longest s to t-1)
            else if (s[i-1] != t[j-1]) dp[i][j]=max(dp[i-1][j], dp[i][j-1]);
        }
    }
    return dp[s.size()][t.size()];
}

// Helper function to convert a number to a vector of its digits
std::vector<ll> numberToVector(ll number) {
    std::vector<ll> digits;
    while (number > 0) {
        digits.push_back(number % 10);
        number /= 10;
    }
    std::reverse(digits.begin(), digits.end());
    return digits;
}

// 8120 becomes [0 1 2 8]
template <typename T, std::enable_if_t<std::is_integral_v<T>, std::nullptr_t> = nullptr>
std::vector<T> digits_low_to_high(T val, T base = 10) {
    std::vector<T> res;
    for (; val; val /= base) res.push_back(val % base);
    if (res.empty()) res.push_back(T{ 0 });
    return res;
}

// 8120 becomes [8 1 2 0]
template <typename T, std::enable_if_t<std::is_integral_v<T>, std::nullptr_t> = nullptr>
std::vector<T> digits_high_to_low(T val, T base = 10) {
    auto res = digits_low_to_high(val, base);
    std::reverse(res.begin(), res.end());
    return res;
}

// Helper function to convert a vector of digits back to a number
ll vectorToNumber(const std::vector<ll>& digits) {
    ll number = 0;
    for (ll digit : digits) {
        number = number * 10 + digit;
    }
    return number;
}

//checks whether vec1 is lexicographically smaller than vec2
bool isLexicographicallySmaller(const std::vector<long long>& vec1, const std::vector<long long>& vec2) {
    return std::lexicographical_compare(vec1.begin(), vec1.end(), vec2.begin(), vec2.end());
}

// cout all permutations of a vector<ll> in lexicographic order
void lexperm(vector<ll> vec){
    while (std::next_permutation(vec.begin(), vec.end())){
        // Print the current permutation
        for (ll num : vec) {
            std::cout << num << " ";
        }
        std::cout << "\n";
    }
}

bool isPalindrome(const std::string& s) {
    int start = 0;
    int end = s.length() - 1;

    while(start < end) {
        // Skip non-alphanumeric characters
        while(start < end && !isalnum(s[start])) start++;
        while(start < end && !isalnum(s[end])) end--;

        // Check for palindrome, ignoring case
        if(tolower(s[start]) != tolower(s[end])) {
            return false;
        }

        start++;
        end--;
    }

    return true;
}

bool isPalindrome(long long n) {
    if (n < 0) return false; // Negative numbers are not considered palindromes

    long long reversed = 0, original = n, remainder;

    while (n != 0) {
        remainder = n % 10;
        reversed = reversed * 10 + remainder;
        n /= 10;
    }

    return original == reversed;
}

//max heap priority queue i.e. top() gives largest value
typedef priority_queue<ll> maxpq;
//min heap priority queue i.e. top() gives smallest value
typedef priority_queue<ll, vector<ll>, greater<ll>> minpq;

//multiset provides automatic ordering on insertion but unlike set, keeps duplicate/multiple items of same value
//n.b. set also provides autoamtic ordering on insertion
//.count(x) O(num_of_x+logN)
//.find(x) O(logN) -> so use find over count if possible
//.insert(x) O(logN) -> inserts s.t. sorted order is maintained
//.erase(x) O(logN)
//begin() O(logN)
typedef multiset<ll> msll;
//doing mymultiset.erase(x) will erase all
#define mserasesingle(mymultiset, x) mymultiset.erase(mymultiset.find(x))
#define mseraseall(mymultiset, x) mymultiset.erase(x)
//find smallest and biggest elements O(1)
#define msmin(mymultiset) *mymultiset.begin()
#define msmax(mymultiset) *mymultiset.rbegin()

int digit_to_int(char c) { return c - '0'; }
int lowercase_to_int(char c) { return c - 'a'; }
int uppercase_to_int(char c) { return c - 'A'; }


template <typename Func, typename Seq>
auto transform_to_vector(const Func &f, const Seq &s) {
    std::vector<std::invoke_result_t<Func, typename Seq::value_type>> v;
    v.reserve(std::size(s)), std::transform(std::begin(s), std::end(s), std::back_inserter(v), f);
    return v;
}
template <typename T, typename Seq>
auto copy_to_vector(const Seq &s) {
    std::vector<T> v;
    v.reserve(std::size(s)), std::copy(std::begin(s), std::end(s), std::back_inserter(v));
    return v;
}
template <typename Seq>
Seq concat(Seq s, const Seq &t) {
    s.reserve(std::size(s) + std::size(t));
    std::copy(std::begin(t), std::end(t), std::back_inserter(s));
    return s;
}
std::vector<int> digit_str_to_ints(const std::string &s) {
    return transform_to_vector(digit_to_int, s);
}
std::vector<int> lowercase_str_to_ints(const std::string &s) {
    return transform_to_vector(lowercase_to_int, s);
}
std::vector<int> uppercase_str_to_ints(const std::string &s) {
    return transform_to_vector(uppercase_to_int, s);
}
template <typename Seq>
std::vector<Seq> split(const Seq s, typename Seq::value_type delim) {
    std::vector<Seq> res;
    for (auto itl = std::begin(s), itr = itl;; itl = ++itr) {
        while (itr != std::end(s) and *itr != delim) ++itr;
        res.emplace_back(itl, itr);
        if (itr == std::end(s)) return res;
    }
}
// Overload of split function to handle C-style strings
std::vector<std::string> split(const char* s, char delim) {
    return split(std::string(s), delim);
}


#include <functional>

#if __cplusplus >= 202002L
#include <bit>
#endif

namespace internal {

#if __cplusplus >= 202002L

using std::bit_ceil;

#else

// @return same with std::bit::bit_ceil
unsigned int bit_ceil(unsigned int n) {
    unsigned int x = 1;
    while (x < (unsigned int)(n)) x *= 2;
    return x;
}

#endif

// @param n `1 <= n`
// @return same with std::bit::countr_zero
int countr_zero(unsigned int n) {
#ifdef _MSC_VER
    unsigned long index;
    _BitScanForward(&index, n);
    return index;
#else
    return __builtin_ctz(n);
#endif
}

// @param n `1 <= n`
// @return same with std::bit::countr_zero
constexpr int countr_zero_constexpr(unsigned int n) {
    int x = 0;
    while (!(n & (1 << x))) x++;
    return x;
}

}  // namespace internal




#if __cplusplus >= 201703L

template <class S,
          auto op,
          auto e,
          class F,
          auto mapping,
          auto composition,
          auto id>
struct lazy_segtree {
    static_assert(std::is_convertible_v<decltype(op), std::function<S(S, S)>>,
                  "op must work as S(S, S)");
    static_assert(std::is_convertible_v<decltype(e), std::function<S()>>,
                  "e must work as S()");
    static_assert(
        std::is_convertible_v<decltype(mapping), std::function<S(F, S)>>,
        "mapping must work as F(F, S)");
    static_assert(
        std::is_convertible_v<decltype(composition), std::function<F(F, F)>>,
        "compostiion must work as F(F, F)");
    static_assert(std::is_convertible_v<decltype(id), std::function<F()>>,
                  "id must work as F()");

#else

template <class S,
          S (*op)(S, S),
          S (*e)(),
          class F,
          S (*mapping)(F, S),
          F (*composition)(F, F),
          F (*id)()>
struct lazy_segtree {

#endif

  public:
    lazy_segtree() : lazy_segtree(0) {}
    explicit lazy_segtree(int n) : lazy_segtree(std::vector<S>(n, e())) {}
    explicit lazy_segtree(const std::vector<S>& v) : _n(int(v.size())) {
        size = (int)internal::bit_ceil((unsigned int)(_n));
        log = internal::countr_zero((unsigned int)size);
        d = std::vector<S>(2 * size, e());
        lz = std::vector<F>(size, id());
        for (int i = 0; i < _n; i++) d[size + i] = v[i];
        for (int i = size - 1; i >= 1; i--) {
            update(i);
        }
    }

    void set(int p, S x) {
        assert(0 <= p && p < _n);
        p += size;
        for (int i = log; i >= 1; i--) push(p >> i);
        d[p] = x;
        for (int i = 1; i <= log; i++) update(p >> i);
    }

    S get(int p) {
        assert(0 <= p && p < _n);
        p += size;
        for (int i = log; i >= 1; i--) push(p >> i);
        return d[p];
    }

    S prod(int l, int r) {
        assert(0 <= l && l <= r && r <= _n);
        if (l == r) return e();

        l += size;
        r += size;

        for (int i = log; i >= 1; i--) {
            if (((l >> i) << i) != l) push(l >> i);
            if (((r >> i) << i) != r) push((r - 1) >> i);
        }

        S sml = e(), smr = e();
        while (l < r) {
            if (l & 1) sml = op(sml, d[l++]);
            if (r & 1) smr = op(d[--r], smr);
            l >>= 1;
            r >>= 1;
        }

        return op(sml, smr);
    }

    S all_prod() { return d[1]; }

    void apply(int p, F f) {
        assert(0 <= p && p < _n);
        p += size;
        for (int i = log; i >= 1; i--) push(p >> i);
        d[p] = mapping(f, d[p]);
        for (int i = 1; i <= log; i++) update(p >> i);
    }
    void apply(int l, int r, F f) {
        assert(0 <= l && l <= r && r <= _n);
        if (l == r) return;

        l += size;
        r += size;

        for (int i = log; i >= 1; i--) {
            if (((l >> i) << i) != l) push(l >> i);
            if (((r >> i) << i) != r) push((r - 1) >> i);
        }

        {
            int l2 = l, r2 = r;
            while (l < r) {
                if (l & 1) all_apply(l++, f);
                if (r & 1) all_apply(--r, f);
                l >>= 1;
                r >>= 1;
            }
            l = l2;
            r = r2;
        }

        for (int i = 1; i <= log; i++) {
            if (((l >> i) << i) != l) update(l >> i);
            if (((r >> i) << i) != r) update((r - 1) >> i);
        }
    }

    template <bool (*g)(S)> int max_right(int l) {
        return max_right(l, [](S x) { return g(x); });
    }
    template <class G> int max_right(int l, G g) {
        assert(0 <= l && l <= _n);
        assert(g(e()));
        if (l == _n) return _n;
        l += size;
        for (int i = log; i >= 1; i--) push(l >> i);
        S sm = e();
        do {
            while (l % 2 == 0) l >>= 1;
            if (!g(op(sm, d[l]))) {
                while (l < size) {
                    push(l);
                    l = (2 * l);
                    if (g(op(sm, d[l]))) {
                        sm = op(sm, d[l]);
                        l++;
                    }
                }
                return l - size;
            }
            sm = op(sm, d[l]);
            l++;
        } while ((l & -l) != l);
        return _n;
    }

    template <bool (*g)(S)> int min_left(int r) {
        return min_left(r, [](S x) { return g(x); });
    }
    template <class G> int min_left(int r, G g) {
        assert(0 <= r && r <= _n);
        assert(g(e()));
        if (r == 0) return 0;
        r += size;
        for (int i = log; i >= 1; i--) push((r - 1) >> i);
        S sm = e();
        do {
            r--;
            while (r > 1 && (r % 2)) r >>= 1;
            if (!g(op(d[r], sm))) {
                while (r < size) {
                    push(r);
                    r = (2 * r + 1);
                    if (g(op(d[r], sm))) {
                        sm = op(d[r], sm);
                        r--;
                    }
                }
                return r + 1 - size;
            }
            sm = op(d[r], sm);
        } while ((r & -r) != r);
        return 0;
    }

  private:
    int _n, size, log;
    std::vector<S> d;
    std::vector<F> lz;

    void update(int k) { d[k] = op(d[2 * k], d[2 * k + 1]); }
    void all_apply(int k, F f) {
        d[k] = mapping(f, d[k]);
        if (k < size) lz[k] = composition(f, lz[k]);
    }
    void push(int k) {
        all_apply(2 * k, lz[k]);
        all_apply(2 * k + 1, lz[k]);
        lz[k] = id();
    }
};

//h S represents a node in the tree i.e. a segment in the original array
// struct S {
//     int upper, lower;
// };

// using F = int;

//h defines how to merge 2 segments of the tree to build up the tree & to query a range
// S op(S l, S r) { return S{l.upper + r.upper, l.lower + r.lower}; }

//h identity element for operation that combines segments
//h segment that doesn't change anything when combined with another segment "identity element". Needed for things like initialization or when updating bits of segment tree that don't need to be changed
//h e comes from German Einheit = unit/unity i.e. unifying/neutral element
//h in this case, the function initializes S upper and lower to 0 and 0
// S e() { return S{0, 0}; }

//h defines how a function/operation is applied to a segment
// S mapping(F l, S r) {
//     if(l == 0) {
//         return r;
//     } else if(l == 2) {
//         return S{0, r.upper + r.lower};
//     } else if(l == 3) {
//         return S{r.upper + r.lower, 0};
//     }
// }

//h defines how to combine 2 updates into one
// F composition(F l, F r) { return l ? l : r; }

//h identity element for operation that updates elements
// F id() { return 0; }

// int main() {
//     int N, Q;
//     string s;
//     cin >> N >> s >> Q;
    
//     vector<S> v(N);
//     for (int i = 0; i < N; ++i) {
//         v[i] = islower(s[i]) ? S{0, 1} : S{1, 0};
//     }

//     lazy_segtree<S, op, e, F, mapping, composition, id> seg(v);
//     for(int _ = 0; _ < Q; ++_) {
//         int t, x;
//         char c;
//         cin >> t >> x >> c, --x;
//         if (t == 1) {
//             seg.apply(x, islower(c) ? 2 : 3);
//             s[x] = c;
//         } else if (t == 2) {
//             seg.apply(0, N, 2);
//         } else {
//             seg.apply(0, N, 3);
//         }
//     }

//     for(int i = 0; i < N; ++i) {
//         s[i] = seg.get(i).lower ? tolower(s[i]) : toupper(s[i]);
//     }
//     cout << s << endl;
// }


#if __cplusplus >= 201703L

template <class S, auto op, auto e> struct segtree {
    static_assert(std::is_convertible_v<decltype(op), std::function<S(S, S)>>,
                  "op must work as S(S, S)");
    static_assert(std::is_convertible_v<decltype(e), std::function<S()>>,
                  "e must work as S()");

#else

template <class S, S (*op)(S, S), S (*e)()> struct segtree {
//initialize as follows:
// struct S {
//     int a;
// };
// S op(S l, S r){
//     return S{max(l.a,r.a)};
// }
// S e(){
//     return S{-1};
// }
// segtree<S, op, e> mysegtree(a);
#endif

  public:
    segtree() : segtree(0) {}
    explicit segtree(int n) : segtree(std::vector<S>(n, e())) {}
    explicit segtree(const std::vector<S>& v) : _n(int(v.size())) {
        size = (int)internal::bit_ceil((unsigned int)(_n));
        log = internal::countr_zero((unsigned int)size);
        d = std::vector<S>(2 * size, e());
        for (int i = 0; i < _n; i++) d[size + i] = v[i];
        for (int i = size - 1; i >= 1; i--) {
            update(i);
        }
    }
    //update value at position p with value x
    //remember 0-indexed so will likely have to do p--
    void set(int p, S x) {
        assert(0 <= p && p < _n);
        p += size;
        d[p] = x;
        for (int i = 1; i <= log; i++) update(p >> i);
    }

    S get(int p) const {
        assert(0 <= p && p < _n);
        return d[p + size];
    }

    //query seg tree in range l->r
    //remember 0-indexed so will likely have to do l-- (note r not inclusive i.e. < rather than <= so no need to do r--)
    S prod(int l, int r) const {
        assert(0 <= l && l <= r && r <= _n);
        S sml = e(), smr = e();
        l += size;
        r += size;

        while (l < r) {
            if (l & 1) sml = op(sml, d[l++]);
            if (r & 1) smr = op(d[--r], smr);
            l >>= 1;
            r >>= 1;
        }
        return op(sml, smr);
    }

    S all_prod() const { return d[1]; }

    template <bool (*f)(S)> int max_right(int l) const {
        return max_right(l, [](S x) { return f(x); });
    }
    //binary search. Initially considers aggregate of segment from l to end of array
    //looks for FIRST/leftmost index r where condition given by f transitions from true to false
    //i.e. returns left-most index where condition false
    //n.b. can use lambda for f e.g.:
    //cout << mysegtree.max_right(x,[&](S b){return b.a<v;})+1<<endl;
    template <class F> int max_right(int l, F f) const {
        assert(0 <= l && l <= _n);
        assert(f(e()));
        if (l == _n) return _n;
        l += size;
        S sm = e();
        do {
            while (l % 2 == 0) l >>= 1;
            if (!f(op(sm, d[l]))) {
                while (l < size) {
                    l = (2 * l);
                    if (f(op(sm, d[l]))) {
                        sm = op(sm, d[l]);
                        l++;
                    }
                }
                return l - size;
            }
            sm = op(sm, d[l]);
            l++;
        } while ((l & -l) != l);
        return _n;
    }

    template <bool (*f)(S)> int min_left(int r) const {
        return min_left(r, [](S x) { return f(x); });
    }
    template <class F> int min_left(int r, F f) const {
        assert(0 <= r && r <= _n);
        assert(f(e()));
        if (r == 0) return 0;
        r += size;
        S sm = e();
        do {
            r--;
            while (r > 1 && (r % 2)) r >>= 1;
            if (!f(op(d[r], sm))) {
                while (r < size) {
                    r = (2 * r + 1);
                    if (f(op(d[r], sm))) {
                        sm = op(d[r], sm);
                        r--;
                    }
                }
                return r + 1 - size;
            }
            sm = op(d[r], sm);
        } while ((r & -r) != r);
        return 0;
    }

  private:
    int _n, size, log;
    std::vector<S> d;

    void update(int k) { d[k] = op(d[2 * k], d[2 * k + 1]); }
};


#include <type_traits>


namespace internal {

#ifndef _MSC_VER
template <class T>
using is_signed_int128 =
    typename std::conditional<std::is_same<T, __int128_t>::value ||
                                  std::is_same<T, __int128>::value,
                              std::true_type,
                              std::false_type>::type;

template <class T>
using is_unsigned_int128 =
    typename std::conditional<std::is_same<T, __uint128_t>::value ||
                                  std::is_same<T, unsigned __int128>::value,
                              std::true_type,
                              std::false_type>::type;

template <class T>
using make_unsigned_int128 =
    typename std::conditional<std::is_same<T, __int128_t>::value,
                              __uint128_t,
                              unsigned __int128>;

template <class T>
using is_integral = typename std::conditional<std::is_integral<T>::value ||
                                                  is_signed_int128<T>::value ||
                                                  is_unsigned_int128<T>::value,
                                              std::true_type,
                                              std::false_type>::type;

template <class T>
using is_signed_int = typename std::conditional<(is_integral<T>::value &&
                                                 std::is_signed<T>::value) ||
                                                    is_signed_int128<T>::value,
                                                std::true_type,
                                                std::false_type>::type;

template <class T>
using is_unsigned_int =
    typename std::conditional<(is_integral<T>::value &&
                               std::is_unsigned<T>::value) ||
                                  is_unsigned_int128<T>::value,
                              std::true_type,
                              std::false_type>::type;

template <class T>
using to_unsigned = typename std::conditional<
    is_signed_int128<T>::value,
    make_unsigned_int128<T>,
    typename std::conditional<std::is_signed<T>::value,
                              std::make_unsigned<T>,
                              std::common_type<T>>::type>::type;

#else

template <class T> using is_integral = typename std::is_integral<T>;

template <class T>
using is_signed_int =
    typename std::conditional<is_integral<T>::value && std::is_signed<T>::value,
                              std::true_type,
                              std::false_type>::type;

template <class T>
using is_unsigned_int =
    typename std::conditional<is_integral<T>::value &&
                                  std::is_unsigned<T>::value,
                              std::true_type,
                              std::false_type>::type;

template <class T>
using to_unsigned = typename std::conditional<is_signed_int<T>::value,
                                              std::make_unsigned<T>,
                                              std::common_type<T>>::type;

#endif

template <class T>
using is_signed_int_t = std::enable_if_t<is_signed_int<T>::value>;

template <class T>
using is_unsigned_int_t = std::enable_if_t<is_unsigned_int<T>::value>;

template <class T> using to_unsigned_t = typename to_unsigned<T>::type;

}  // namespace internal


template <class T> struct fenwick_tree {
    using U = internal::to_unsigned_t<T>;

  public:
    // Declare fenwick_tree (N elements initialized to 0) by doing:
    // fenwick_tree< long  long > ft (N);
    // fenwick tree is held in a vector<T> called data
    fenwick_tree() : _n(0) {}
    explicit fenwick_tree(int n) : _n(n), data(n) {}

    // a is 0-indexed
    // use add to add array item 'x' to index 'a' in Fenwick tree
    // n.b. index 'a' in Fenwick tree represents a range of responsibility
    // i.e. holds a prefix sum for a particular range of original array
    // this range of responsibility is determined by index 'a's binary representation
    // it is responsible for E elements below it
    // where E is the index of its LSB where index is from R->L of binary number
    // e.g. 11010 LSB index is 2
    void add(int p, T x) {
        assert(0 <= p && p < _n);
        p++;
        while (p <= _n) {
            data[p - 1] += U(x);
            p += p & -p;
        }
    }

    // Get sum over range [l, r), where l is 0-indexed
    T sum(int l, int r) {
        assert(0 <= l && l <= r && r <= _n);
        return sum(r) - sum(l);
    }

  private:
    int _n;
    std::vector<U> data;

    U sum(int r) {
        U s = 0;
        while (r > 0) {
            s += data[r - 1];
            r -= r & -r;
        }
        return s;
    }
};


//Use fenwick_tree above instead
template <class T> struct BIT {
    T UNITY_SUM = 0;
    vector<T> dat;
    
    // [0, n)
    // Declare BIT (N elements initialized to 0) by doing:
    // BIT< long  long > bit (N);
    // fenwick tree is held in a vector<T> called dat
    BIT(int n, T unity = 0) : UNITY_SUM(unity), dat(n, unity) { }
    
    //allows reinitialization of the tree resetting all elements to unity sum
    void init(int n) {
        dat.assign(n, UNITY_SUM);
    }
    
    // a is 0-indexed
    // use add to add array item 'x' to index 'a' in Fenwick tree
    // n.b. index 'a' in Fenwick tree represents a range of responsibility
    // i.e. holds a prefix sum for a particular range of original array
    // this range of responsibility is determined by index 'a's binary representation
    // it is responsible for E elements below it
    // where E is the index of its LSB where index is from R->L of binary number
    // e.g. 11010 LSB index is 2
    inline void add(int a, T x) {
        for (int i = a; i < (int)dat.size(); i |= i + 1)
            dat[i] = dat[i] + x;
    }
    
    // Get sum over range [0, a), where a is 0-indexed
    inline T sum(int a) {
        T res = UNITY_SUM;
        for (int i = a - 1; i >= 0; i = (i & (i + 1)) - 1)
            res = res + dat[i];
        return res;
    }
    
    // Get sum over range [a, b), where a and b are 0-indexed
    inline T sum(int a, int b) {
        return sum(b) - sum(a);
    }
    
    // debug
    // prints the values of original array after modifications
    void print() {
        for (int i = 0; i < (int)dat.size(); ++i)
            cerr << sum(i, i + 1) << ",";
        cerr << endl;
    }
};

//for iterating over possible directions from a square in a 2d array -> for both wasd & including diagonals
vector<int> dx = {1, 0, -1, 0, 1, 1, -1, -1};
vector<int> dx_wasd = {1,-1,0,0};
vector<int> dy = {0, 1, 0, -1, 1, -1, 1, -1};
vector<int> dy_wasd = {0,0,1,-1};

//Graph visualizer:
//https://csacademy.com/app/graph_editor/

constexpr long long MOD = 1000000007;
long long solve(long long N) {
    /* vis.assign(n+1, false);
    g.assign(n+1, vector<int>());
    wg.assign(n + 1, vector<pair<ll,ll>>());
    parent.assign(n+1, -1); */
}

int main() {
    std::ios::sync_with_stdio(false);
    setIO("");
    std::cin.tie(nullptr);
    long long N;
    std::cin >> N;
    auto ans = solve(N);
    std::cout << ans << '\n';

    /* genprimes(1e5); */

    /* //run the bfs and output order of traversed nodes (for loop is only used for non-connected graphs)
    for (int i = 0; i < n; i++) {
        if (!v[i])
            bfs(i);
    }
    
    //Use for problems where you have to go up,down,left,right. Do x+i & y+j and i&j will test all 4 directions. Do x+i+1 & y+j+1 if 0 indexed
    wasd(
        //cout << "Use this for problems where you have to go up, down, left right" << endl;
    ) */
    return 0;
}
