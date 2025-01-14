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
#include <cassert>
#include <unordered_map>
#include <type_traits> // For std::is_floating_point
#include <cmath> // For std::ceil
#include <iomanip>
#include <unordered_set>
#include <functional>
#include <type_traits>
#include <chrono>

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
// #define isvalid(x_plus_i,max_boundary_n,y_plus_j,max_boundary_m) (0<=x_plus_i and x_plus_i<max_boundary_n and 0<=y_plus_j and y_plus_j<max_boundary_m)
#define isvalid(x_value,y_value,min_valid_x,max_valid_x,min_valid_y,max_valid_y) (min_valid_x<=x_value and x_value<=max_valid_x and min_valid_y<=y_value and y_value<=max_valid_y)
//#define gcd __gcd
#define mp make_pair
#define mt make_tuple
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
#define binstoll(binaryStringToConvertToInt) stoll(binaryStringToConvertToInt, nullptr, 2)
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
ll INF=LLONG_MAX;


// gp_hash_table<long long, int, custom_hash> safe_hash_table;


struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        // http://xorshift.di.unimi.it/splitmix64.c
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};

/*/---------------------------IO(Debugging)----------------------/*/
template<class T> istream& operator >> (istream &is, vector<T>& V) {
    for(auto &e : V)
        is >> e;
    return is;
}
template <class OStream, class T> OStream &operator<<(OStream &os, const std::vector<T> &vec);
template <class OStream, class T, size_t sz> OStream &operator<<(OStream &os, const std::array<T, sz> &arr);
template <class OStream, class T, class TH> OStream &operator<<(OStream &os, const std::unordered_set<T, TH> &vec);
template <class OStream, class T, class U> OStream &operator<<(OStream &os, const pair<T, U> &pa);
template <class OStream, class T> OStream &operator<<(OStream &os, const std::deque<T> &vec);
template <class OStream, class T> OStream &operator<<(OStream &os, const std::set<T> &vec);
template <class OStream, class T> OStream &operator<<(OStream &os, const std::multiset<T> &vec);
template <class OStream, class T> OStream &operator<<(OStream &os, const std::unordered_multiset<T> &vec);
template <class OStream, class T, class U> OStream &operator<<(OStream &os, const std::pair<T, U> &pa);
template <class OStream, class TK, class TV> OStream &operator<<(OStream &os, const std::map<TK, TV> &mp);
template <class OStream, class TK, class TV, class TH> OStream &operator<<(OStream &os, const std::unordered_map<TK, TV, TH> &mp);
template <class OStream, class... T> OStream &operator<<(OStream &os, const std::tuple<T...> &tpl);

template <class OStream, class T> OStream &operator<<(OStream &os, const std::vector<T> &vec) { os << '['; for (auto v : vec) os << v << ','; os << ']'; return os; }
template <class OStream, class T, size_t sz> OStream &operator<<(OStream &os, const std::array<T, sz> &arr) { os << '['; for (auto v : arr) os << v << ','; os << ']'; return os; }
template <class... T> std::istream &operator>>(std::istream &is, std::tuple<T...> &tpl) { std::apply([&is](auto &&... args) { ((is >> args), ...);}, tpl); return is; }
template <class OStream, class... T> OStream &operator<<(OStream &os, const std::tuple<T...> &tpl) { os << '('; std::apply([&os](auto &&... args) { ((os << args << ','), ...);}, tpl); return os << ')'; }
template <class OStream, class T, class TH> OStream &operator<<(OStream &os, const std::unordered_set<T, TH> &vec) { os << '{'; for (auto v : vec) os << v << ','; os << '}'; return os; }
template <class OStream, class T> OStream &operator<<(OStream &os, const std::deque<T> &vec) { os << "deq["; for (auto v : vec) os << v << ','; os << ']'; return os; }
template <class OStream, class T> OStream &operator<<(OStream &os, const std::set<T> &vec) { os << '{'; for (auto v : vec) os << v << ','; os << '}'; return os; }
template <class OStream, class T> OStream &operator<<(OStream &os, const std::multiset<T> &vec) { os << '{'; for (auto v : vec) os << v << ','; os << '}'; return os; }
template <class OStream, class T> OStream &operator<<(OStream &os, const std::unordered_multiset<T> &vec) { os << '{'; for (auto v : vec) os << v << ','; os << '}'; return os; }
template <class OStream, class T, class U> OStream &operator<<(OStream &os, const std::pair<T, U> &pa) { return os << '(' << pa.first << ',' << pa.second << ')'; }
template <class OStream, class TK, class TV> OStream &operator<<(OStream &os, const std::map<TK, TV> &mp) { os << '{'; for (auto v : mp) os << v.first << "=>" << v.second << ','; os << '}'; return os; }
template <class OStream, class TK, class TV, class TH> OStream &operator<<(OStream &os, const std::unordered_map<TK, TV, TH> &mp) { os << '{'; for (auto v : mp) os << v.first << "=>" << v.second << ','; os << '}'; return os; }


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

//actually make the list of primes only not just the list of all numbers indicating which are prime and which are not
vll GenListOfPrimesOnly(){
    gen_primes();
    vll allprimes;
    for(int i = 0; i<MAXA; i++){
        if(prime[i]==1){
            allprimes.pb(i);
        }
    }
    return allprimes;
}

bool IsPrime(int num)
{
    if (num < 2) return false;
    else if (num == 2) return true;
    else if (num % 2 == 0) return false; // 偶数はあらかじめ除く

    double sqrtNum = sqrt(num);
    for (int i = 3; i <= sqrtNum; i += 2)
    {
        if (num % i == 0)
        {
            // 素数ではない
            return false;
        }
    }
    return true;
}

// To get prime factorization:
// SieveOfEratosthenes sieve(1000000);
// auto facs = sieve.Factorize(N);
struct SieveOfEratosthenes : std::vector<int>
{
    std::vector<int> primes;
    SieveOfEratosthenes(int MAXN) : std::vector<int>(MAXN + 1) {
        std::iota(begin(), end(), 0);
        for (int i = 2; i <= MAXN; i++) {
            if ((*this)[i] == i) {
                primes.push_back(i);
                for (int j = i; j <= MAXN; j += i) (*this)[j] = i;
            }
        }
    }
    using T = long long int;
    // Prime factorization for x <= MAXN^2
    // Complexity: O(log x)          (x <= MAXN)
    //             O(MAXN / logMAXN) (MAXN < x <= MAXN^2)
    std::map<T, int> Factorize(T x) {
        assert(x <= 1LL * (int(size()) - 1) * (int(size()) - 1));
        std::map<T, int> ret;
        if (x < int(size())) {
            while (x > 1) {
                ret[(*this)[x]]++;
                x /= (*this)[x];
            }
        }
        else {
            for (auto p : primes) {
                while (!(x % p)) x /= p, ret[p]++;
                if (x == 1) break;
            }
            if (x > 1) ret[x]++;
        }
        return ret;
    }
    std::vector<T> Divisors(T x) {
        std::vector<T> ret{1};
        for (auto p : Factorize(x)) {
            int n = ret.size();
            for (int i = 0; i < n; i++) {
                for (T a = 1, d = 1; d <= p.second; d++) {
                    a *= p.first;
                    ret.push_back(ret[i] * a);
                }
            }
        }
        return ret; // Not sorted
    }
    // Moebius function Table
    // return: [0=>0, 1=>1, 2=>-1, 3=>-1, 4=>0, 5=>-1, 6=>1, 7=>-1, 8=>0, ...]
    std::vector<int> GenerateMoebiusFunctionTable() {
        std::vector<int> ret(size());
        for (int i = 1; i < int(size()); i++) {
            if (i == 1) ret[i] = 1;
            else if ((i / (*this)[i]) % (*this)[i] == 0) ret[i] = 0;
            else ret[i] = -ret[i / (*this)[i]];
        }
        return ret;
    }
};
SieveOfEratosthenes sieve(1000000);



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
    g[destNode].pb(originNode);
}

void edge(ll originNode, ll destNode, ll weight){
    wg[originNode].emplace_back(destNode, weight);
    totalEdges++;
    // For an undirected graph e.g., tree, add this line:
    wg[destNode].emplace_back(originNode, weight);
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
//don't forget to initialize parent with parent.assign(n+1, -1);
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
void dddfs(ll startNode, ll startDepth){
    vis[startNode] = true;
    depth[startNode]=startDepth;
    maxDepth=max(maxDepth, startDepth);
    cerr << startNode << " ";
    for(auto adjNode : g[startNode]){
        if(!vis[adjNode]) dddfs(adjNode, startDepth+1);
    }
}

//weighted undirected graph 全探索 to find max Hamiltonian path cost
//https://atcoder.jp/contests/abc317/submissions/51739795
ll ans;
void ddddfs(ll v, vector<bool> vis, ll cost){
    ans = max(ans,cost);
    for(auto x:wg[v]){
        if(!vis[x.first]){
            vis[x.first]=true;
            ddddfs(x.first, vis, cost+x.second);
            vis[x.first]=false;
        }
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

// lowest common ancestor (LCA) class for undirected weighted tree
struct UndirectedWeightedTree {
    using T = long long; // Arbitrary data structure (operator+, operator- must be defined)
    int INVALID = -1;
    int V, lgV;
    int E;
    int root;
    std::vector<std::vector<std::pair<int, int>>> adj; // (nxt_vertex, edge_id)
    // vector<pint> edge; // edges[edge_id] = (vertex_id, vertex_id)
    std::vector<T> weight;     // w[edge_id]
    std::vector<int> par;      // parent_vertex_id[vertex_id]
    std::vector<int> depth;    // depth_from_root[vertex_id]
    std::vector<T> acc_weight; // w_sum_from_root[vertex_id]

    void _fix_root_dfs(int now, int prv, int prv_edge_id) {
        par[now] = prv;
        if (prv_edge_id != INVALID) acc_weight[now] = acc_weight[prv] + weight[prv_edge_id];
        for (auto nxt : adj[now])
            if (nxt.first != prv) {
                depth[nxt.first] = depth[now] + 1;
                _fix_root_dfs(nxt.first, now, nxt.second);
            }
    }

    UndirectedWeightedTree() = default;
    UndirectedWeightedTree(int N) : V(N), E(0), adj(N) {
        lgV = 1;
        while (1 << lgV < V) lgV++;
    }

    void add_edge(int u, int v, T w) {
        adj[u].emplace_back(v, E);
        adj[v].emplace_back(u, E);
        // edge.emplace_back(u, v);
        weight.emplace_back(w);
        E++;
    }

    //Have to set which node you want to consider as root. This function will then run dfs to calculate depth for every node
    void fix_root(int r) {
        root = r;
        par.resize(V);
        depth.resize(V);
        depth[r] = 0;
        acc_weight.resize(V);
        acc_weight[r] = 0;
        _fix_root_dfs(root, INVALID, INVALID);
    }

    //this does binary lifting precalculation -> gives powers of 2 ancestors of every node
    //as every number can be represented in binary, you can find any kth ancestor of any node in logarithmic time if you precalculate above
    std::vector<std::vector<int>> doubling;
    void doubling_precalc() {
        doubling.assign(lgV, std::vector<int>(V));
        doubling[0] = par;
        for (int d = 0; d < lgV - 1; d++)
            for (int i = 0; i < V; i++) {
                if (doubling[d][i] == INVALID)
                    doubling[d + 1][i] = INVALID;
                else
                    doubling[d + 1][i] = doubling[d][doubling[d][i]];
            }
    }

    int kth_parent(int x, int k) {
        if (depth[x] < k) return INVALID;
        for (int d = 0; d < lgV; d++) {
            if (x == INVALID) return INVALID;
            if (k & (1 << d)) x = doubling[d][x];
        }
        return x;
    }

    //returns the lca of 2 nodes
    int lowest_common_ancestor(int u, int v) {
        if (depth[u] > depth[v]) std::swap(u, v);

        v = kth_parent(v, depth[v] - depth[u]);
        if (u == v) return u;
        for (int d = lgV - 1; d >= 0; d--) {
            if (doubling[d][u] != doubling[d][v]) u = doubling[d][u], v = doubling[d][v];
        }
        return par[u];
    }

    //uses LCA to calculate distance between 2 nodes in O(log N) time
    T path_length(int u, int v) {
        // Not distance, but the sum of weights
        int r = lowest_common_ancestor(u, v);
        return (acc_weight[u] - acc_weight[r]) + (acc_weight[v] - acc_weight[r]);
    }
};

vector<vector<ll> >ddist;

void floydWarshall(ll n) {
    // Step 1: Initialization
    ddist.assign(n, vector<ll>(n, INF));
    for (int i = 0; i < n; i++) {
        ddist[i][i] = 0;
    }

    // Step 2: Populate the adjacency matrix
    for (int u = 0; u < n; u++) {
        for (int i = 0; i < g[u].size(); i++) {
            ll v = g[u][i];
            ddist[u][v] = 1; // Assuming unweighted graph, set weight to 1
            // For a weighted graph, you'd set this to the weight of the edge u->v
        }
    }

    // Step 3: Floyd-Warshall Algorithm
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (ddist[i][k] < INF &&ddist[k][j] < INF) {
                    ddist[i][j] = min(ddist[i][j],ddist[i][k] +ddist[k][j]);
                }
            }
        }
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

string itobins(ll n) {
    if (n == 0) return "0";

    string binary = "";
    while (n > 0) {
        binary += (n % 2) ? '1' : '0';
        n /= 2;
    }

    reverse(binary.begin(), binary.end()); // Reverse to get the correct order
    return binary;
}

string dtobx(ll decimalNumber, ll base) {
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

#if __cplusplus >= 202002L
#include <bit>
#endif
namespace internal {
#if __cplusplus >= 202002L
using std::bit_ceil;
#else
unsigned int bit_ceil(unsigned int n) {
    unsigned int x = 1;
    while (x < (unsigned int)(n)) x *= 2;
    return x;
}

#endif
int countr_zero(unsigned int n) {
#ifdef _MSC_VER
    unsigned long index;
    _BitScanForward(&index, n);
    return index;
#else
    return __builtin_ctz(n);
#endif
}
constexpr int countr_zero_constexpr(unsigned int n) {
    int x = 0;
    while (!(n & (1 << x))) x++;
    return x;
}

}  // namespace internal
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

namespace internal {

// @param m `1 <= m`
// @return x mod m
constexpr long long safe_mod(long long x, long long m) {
    x %= m;
    if (x < 0) x += m;
    return x;
}

// Fast modular multiplication by barrett reduction
// Reference: https://en.wikipedia.org/wiki/Barrett_reduction
// NOTE: reconsider after Ice Lake
struct barrett {
    unsigned int _m;
    unsigned long long im;

    // @param m `1 <= m`
    explicit barrett(unsigned int m) : _m(m), im((unsigned long long)(-1) / m + 1) {}

    // @return m
    unsigned int umod() const { return _m; }

    // @param a `0 <= a < m`
    // @param b `0 <= b < m`
    // @return `a * b % m`
    unsigned int mul(unsigned int a, unsigned int b) const {
        unsigned long long z = a;
        z *= b;
#ifdef _MSC_VER
        unsigned long long x;
        _umul128(z, im, &x);
#else
        unsigned long long x =
            (unsigned long long)(((unsigned __int128)(z)*im) >> 64);
#endif
        unsigned long long y = x * _m;
        return (unsigned int)(z - y + (z < y ? _m : 0));
    }
};

constexpr long long pow_mod_constexpr(long long x, long long n, int m) {
    if (m == 1) return 0;
    unsigned int _m = (unsigned int)(m);
    unsigned long long r = 1;
    unsigned long long y = safe_mod(x, m);
    while (n) {
        if (n & 1) r = (r * y) % _m;
        y = (y * y) % _m;
        n >>= 1;
    }
    return r;
}

// Reference:
// M. Forisek and J. Jancina,
// Fast Primality Testing for Integers That Fit into a Machine Word
// @param n `0 <= n`
constexpr bool is_prime_constexpr(int n) {
    if (n <= 1) return false;
    if (n == 2 || n == 7 || n == 61) return true;
    if (n % 2 == 0) return false;
    long long d = n - 1;
    while (d % 2 == 0) d /= 2;
    constexpr long long bases[3] = {2, 7, 61};
    for (long long a : bases) {
        long long t = d;
        long long y = pow_mod_constexpr(a, t, n);
        while (t != n - 1 && y != 1 && y != n - 1) {
            y = y * y % n;
            t <<= 1;
        }
        if (y != n - 1 && t % 2 == 0) {
            return false;
        }
    }
    return true;
}
template <int n> constexpr bool is_prime = is_prime_constexpr(n);

constexpr std::pair<long long, long long> inv_gcd(long long a, long long b) {
    a = safe_mod(a, b);
    if (a == 0) return {b, 0};
    long long s = b, t = a;
    long long m0 = 0, m1 = 1;

    while (t) {
        long long u = s / t;
        s -= t * u;
        m0 -= m1 * u;  // |m1 * u| <= |m1| * s <= b

        auto tmp = s;
        s = t;
        t = tmp;
        tmp = m0;
        m0 = m1;
        m1 = tmp;
    }

    if (m0 < 0) m0 += b / s;
    return {s, m0};
}

// Compile time primitive root
// @param m must be prime
// @return primitive root (and minimum in now)
constexpr int primitive_root_constexpr(int m) {
    if (m == 2) return 1;
    if (m == 167772161) return 3;
    if (m == 469762049) return 3;
    if (m == 754974721) return 11;
    if (m == 998244353) return 3;
    int divs[20] = {};
    divs[0] = 2;
    int cnt = 1;
    int x = (m - 1) / 2;
    while (x % 2 == 0) x /= 2;
    for (int i = 3; (long long)(i)*i <= x; i += 2) {
        if (x % i == 0) {
            divs[cnt++] = i;
            while (x % i == 0) {
                x /= i;
            }
        }
    }
    if (x > 1) {
        divs[cnt++] = x;
    }
    for (int g = 2;; g++) {
        bool ok = true;
        for (int i = 0; i < cnt; i++) {
            if (pow_mod_constexpr(g, (m - 1) / divs[i], m) == 1) {
                ok = false;
                break;
            }
        }
        if (ok) return g;
    }
}
template <int m> constexpr int primitive_root = primitive_root_constexpr(m);

// @param n `n < 2^32`
// @param m `1 <= m < 2^32`
// @return sum_{i=0}^{n-1} floor((ai + b) / m) (mod 2^64)
unsigned long long floor_sum_unsigned(unsigned long long n,
                                      unsigned long long m,
                                      unsigned long long a,
                                      unsigned long long b) {
    unsigned long long ans = 0;
    while (true) {
        if (a >= m) {
            ans += n * (n - 1) / 2 * (a / m);
            a %= m;
        }
        if (b >= m) {
            ans += n * (b / m);
            b %= m;
        }

        unsigned long long y_max = a * n + b;
        if (y_max < m) break;
        // y_max < m * (n + 1)
        // floor(y_max / m) <= n
        n = (unsigned long long)(y_max / m);
        b = (unsigned long long)(y_max % m);
        std::swap(m, a);
    }
    return ans;
}

}  // namespace internal


namespace internal {

struct modint_base {};
struct static_modint_base : modint_base {};

template <class T> using is_modint = std::is_base_of<modint_base, T>;
template <class T> using is_modint_t = std::enable_if_t<is_modint<T>::value>;

}  // namespace internal

template <int m, std::enable_if_t<(1 <= m)>* = nullptr>
struct static_modint : internal::static_modint_base {
    using mint = static_modint;

  public:
    static constexpr int mod() { return m; }
    static mint raw(int v) {
        mint x;
        x._v = v;
        return x;
    }

    static_modint() : _v(0) {}
    template <class T, internal::is_signed_int_t<T>* = nullptr>
    static_modint(T v) {
        long long x = (long long)(v % (long long)(umod()));
        if (x < 0) x += umod();
        _v = (unsigned int)(x);
    }
    template <class T, internal::is_unsigned_int_t<T>* = nullptr>
    static_modint(T v) {
        _v = (unsigned int)(v % umod());
    }

    unsigned int val() const { return _v; }

    mint& operator++() {
        _v++;
        if (_v == umod()) _v = 0;
        return *this;
    }
    mint& operator--() {
        if (_v == 0) _v = umod();
        _v--;
        return *this;
    }
    mint operator++(int) {
        mint result = *this;
        ++*this;
        return result;
    }
    mint operator--(int) {
        mint result = *this;
        --*this;
        return result;
    }

    mint& operator+=(const mint& rhs) {
        _v += rhs._v;
        if (_v >= umod()) _v -= umod();
        return *this;
    }
    mint& operator-=(const mint& rhs) {
        _v -= rhs._v;
        if (_v >= umod()) _v += umod();
        return *this;
    }
    mint& operator*=(const mint& rhs) {
        unsigned long long z = _v;
        z *= rhs._v;
        _v = (unsigned int)(z % umod());
        return *this;
    }
    mint& operator/=(const mint& rhs) { return *this = *this * rhs.inv(); }

    mint operator+() const { return *this; }
    mint operator-() const { return mint() - *this; }

    mint pow(long long n) const {
        assert(0 <= n);
        mint x = *this, r = 1;
        while (n) {
            if (n & 1) r *= x;
            x *= x;
            n >>= 1;
        }
        return r;
    }
    mint inv() const {
        if (prime) {
            assert(_v);
            return pow(umod() - 2);
        } else {
            auto eg = internal::inv_gcd(_v, m);
            assert(eg.first == 1);
            return eg.second;
        }
    }

    friend mint operator+(const mint& lhs, const mint& rhs) {
        return mint(lhs) += rhs;
    }
    friend mint operator-(const mint& lhs, const mint& rhs) {
        return mint(lhs) -= rhs;
    }
    friend mint operator*(const mint& lhs, const mint& rhs) {
        return mint(lhs) *= rhs;
    }
    friend mint operator/(const mint& lhs, const mint& rhs) {
        return mint(lhs) /= rhs;
    }
    friend bool operator==(const mint& lhs, const mint& rhs) {
        return lhs._v == rhs._v;
    }
    friend bool operator!=(const mint& lhs, const mint& rhs) {
        return lhs._v != rhs._v;
    }

  private:
    unsigned int _v;
    static constexpr unsigned int umod() { return m; }
    static constexpr bool prime = internal::is_prime<m>;
};

template <int id> struct dynamic_modint : internal::modint_base {
    using mint = dynamic_modint;

  public:
    static int mod() { return (int)(bt.umod()); }
    static void set_mod(int m) {
        assert(1 <= m);
        bt = internal::barrett(m);
    }
    static mint raw(int v) {
        mint x;
        x._v = v;
        return x;
    }

    dynamic_modint() : _v(0) {}
    template <class T, internal::is_signed_int_t<T>* = nullptr>
    dynamic_modint(T v) {
        long long x = (long long)(v % (long long)(mod()));
        if (x < 0) x += mod();
        _v = (unsigned int)(x);
    }
    template <class T, internal::is_unsigned_int_t<T>* = nullptr>
    dynamic_modint(T v) {
        _v = (unsigned int)(v % mod());
    }

    unsigned int val() const { return _v; }

    mint& operator++() {
        _v++;
        if (_v == umod()) _v = 0;
        return *this;
    }
    mint& operator--() {
        if (_v == 0) _v = umod();
        _v--;
        return *this;
    }
    mint operator++(int) {
        mint result = *this;
        ++*this;
        return result;
    }
    mint operator--(int) {
        mint result = *this;
        --*this;
        return result;
    }

    mint& operator+=(const mint& rhs) {
        _v += rhs._v;
        if (_v >= umod()) _v -= umod();
        return *this;
    }
    mint& operator-=(const mint& rhs) {
        _v += mod() - rhs._v;
        if (_v >= umod()) _v -= umod();
        return *this;
    }
    mint& operator*=(const mint& rhs) {
        _v = bt.mul(_v, rhs._v);
        return *this;
    }
    mint& operator/=(const mint& rhs) { return *this = *this * rhs.inv(); }

    mint operator+() const { return *this; }
    mint operator-() const { return mint() - *this; }

    mint pow(long long n) const {
        assert(0 <= n);
        mint x = *this, r = 1;
        while (n) {
            if (n & 1) r *= x;
            x *= x;
            n >>= 1;
        }
        return r;
    }
    mint inv() const {
        auto eg = internal::inv_gcd(_v, mod());
        assert(eg.first == 1);
        return eg.second;
    }

    friend mint operator+(const mint& lhs, const mint& rhs) {
        return mint(lhs) += rhs;
    }
    friend mint operator-(const mint& lhs, const mint& rhs) {
        return mint(lhs) -= rhs;
    }
    friend mint operator*(const mint& lhs, const mint& rhs) {
        return mint(lhs) *= rhs;
    }
    friend mint operator/(const mint& lhs, const mint& rhs) {
        return mint(lhs) /= rhs;
    }
    friend bool operator==(const mint& lhs, const mint& rhs) {
        return lhs._v == rhs._v;
    }
    friend bool operator!=(const mint& lhs, const mint& rhs) {
        return lhs._v != rhs._v;
    }

  private:
    unsigned int _v;
    static internal::barrett bt;
    static unsigned int umod() { return bt.umod(); }
};
template <int id> internal::barrett dynamic_modint<id>::bt(998244353);

using modint998244353 = static_modint<998244353>;
using modint1000000007 = static_modint<1000000007>;
using modint = dynamic_modint<-1>;

namespace internal {

template <class T>
using is_static_modint = std::is_base_of<internal::static_modint_base, T>;

template <class T>
using is_static_modint_t = std::enable_if_t<is_static_modint<T>::value>;

template <class> struct is_dynamic_modint : public std::false_type {};
template <int id>
struct is_dynamic_modint<dynamic_modint<id>> : public std::true_type {};

template <class T>
using is_dynamic_modint_t = std::enable_if_t<is_dynamic_modint<T>::value>;

}  // namespace internal

// use instead of pow when dealing with large numbers as pow() uses floating point arithmetic, which will give wrong results for large numbers
// this allows it to deal with powering decimal numbers etc
// but if you are powering large integers then use ipow which uses int/long long only for accuracy
long long ipow(long long base, int exp) {
    long long result = 1;
    while (exp > 0) {
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
        base *= base;
    }
    return result;
}


bool isBitSet(int number, int bitPosition) {
    return (number & (1 << bitPosition)) != 0;
}

vector<ll> getSetBitPositions(ll num) {
    std::vector<ll> setBitPositions;
    ll position = 0;

    while (num != 0) {
        if (num & 1) {
            setBitPositions.push_back(position);
        }
        num >>= 1;
        position++;
    }

    return setBitPositions;
}

int countUniqueSubstrings(const string& s) {
    set<string> unique_substrings;
    for (size_t i = 0; i < s.size(); ++i) {
        for (size_t j = 1; j <= s.size() - i; ++j) {
            unique_substrings.insert(s.substr(i, j));
        }
    }
    return unique_substrings.size();
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

//["abc", "def"]
//becomes
//["da", "eb", "fc"]
vector<string> rotate90(const vector<string>& matrix) {
    int n = matrix.size();
    int m = matrix[0].size();
    vector<string> rotated(m, string(n, '.'));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            rotated[j][n - 1 - i] = matrix[i][j];
        }
    }
    return rotated;
}

#ifdef isym444_LOCAL
const string COLOR_RESET = "\033[0m", BRIGHT_GREEN = "\033[1;32m", BRIGHT_RED = "\033[1;31m", BRIGHT_CYAN = "\033[1;36m", NORMAL_CROSSED = "\033[0;9;37m", RED_BACKGROUND = "\033[1;41m", NORMAL_FAINT = "\033[0;2m";
#define dbg(x) std::cerr << BRIGHT_CYAN << #x << COLOR_RESET << " = " << (x) << NORMAL_FAINT << " (L" << __LINE__ << ") " << COLOR_RESET << std::endl
#define dbgif(cond, x) ((cond) ? std::cerr << BRIGHT_CYAN << #x << COLOR_RESET << " = " << (x) << NORMAL_FAINT << " (L" << __LINE__ << ") " << __FILE__ << COLOR_RESET << std::endl : std::cerr)
#else
#define dbg(x) ((void)0)
#define dbgif(cond, x) ((void)0)
#endif

template <class T> std::vector<T> sort_unique(std::vector<T> vec) { sort(vec.begin(), vec.end()), vec.erase(unique(vec.begin(), vec.end()), vec.end()); return vec; }
//index of the first occurrence of x. If x is not present in the vector, it returns the index where x can be inserted while keeping the vector sorted
template <class T> int indlb(const std::vector<T> &v, const T &x) { return std::distance(v.begin(), std::lower_bound(v.begin(), v.end(), x)); }
//index immediately after the last occurrence of x. If x is not present, like the lower bound, it returns the index where x can be inserted to maintain order
template <class T> int indub(const std::vector<T> &v, const T &x) { return std::distance(v.begin(), std::upper_bound(v.begin(), v.end(), x)); }


//for iterating over possible directions from a square in a 2d array -> for both wasd & including diagonals
vector<int> dx = {1, 0, -1, 0, 1, 1, -1, -1};
vector<int> dx_wasd = {1,-1,0,0};
vector<int> dy = {0, 1, 0, -1, 1, -1, 1, -1};
vector<int> dy_wasd = {0,0,1,-1};

//Takes in a point and returns vector<pair<int, int>> containing all points relative to og point that form a square given an initial dx & dy between og_point and one other point n.b. always rotating in same direction
vector<pair<int, int>> generateSquarePoints(int x, int y, int dx, int dy) {
    pair<int, int> p1, p2, p3, p4;
    
    p1 = mp(x, y);
    p2 = mp(p1.first + dx, p1.second + dy);
    p3 = mp(p2.first - dy, p2.second + dx);
    p4 = mp(p3.first - dx, p3.second - dy);
    
    // p1 = mp(x,y);
    // p2 = mp(x+dx,y+dy);
    // p3 = mp(x+dx-dy,y+dy+dx);
    // p4 = mp(x-dy,y+dx);

    vector<pair<int, int>> points = {p1, p2, p3, p4};
    return points;
}

//Graph visualizer:
//https://csacademy.com/app/graph_editor/

//n.b. it is a data type so declare variablesas: mint x;
// to convert any other data type such as int or ll to mint, do: mint(x);
// when you want to access the value of a mint, use x.val()
// e.g. modint998244353 a = modint998244353(x); // `a` now represents `x` modulo 998244353
using mint = modint998244353;


int main(){
    ll A,B,C,D;
    cin >> A >> B >> C >> D;
    A+=(ll)1e9;
    B+=(ll)1e9;
    C+=(ll)1e9;
    D+=(ll)1e9;

    dbg(mt(A,B,C,D));

    long long ans = 0;

    auto calc = [&](ll x, ll y)->long long{
        long long tans = 0;

        // vertical extra
        ll verex = MOD(y,2);

        // horizontal extra
        ll horex = MOD(x,4);

        // blue
        ll hor = x-horex;
        ll ver = y-verex;
        tans+=(hor*ver);

        if(verex){
            tans+=hor;
        }
        
        if(horex){
            if(horex==1){
                tans+=ver*3/2;
                if(verex){
                    tans+=2;
                }
            }
            if(horex==2){
                tans+=(ver*2)*3/2;
                if(verex){
                    tans+=3;
                }
            }
            if(horex==3){
                // tans+=(ver)*3/2;
                // tans+=(ver)*3/2;
                // tans+=(ver)*1/2;
                tans+=(ver*3)*7/6;
                if(verex){
                    tans+=3;
                }
            }
        }
        dbg(tans);
        return tans;
    };

    // green
    ans += calc(C,D);

    // blue
    ans -= calc(A,D);

    // orange
    ans -= calc(C,B);
    
    // red
    ans+= calc(A,B);
    // dbg(ans);
    // ll finans = (ll)ans;

    cout << ans << endl;

    return 0;
}