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
#define print(p) cout<<p<<endl
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
struct dsu {
  public:
    dsu() : _n(0) {}
    //constructor for dsu. Initialize as "dsu name_of_object(x);"
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

//Graph visualizer:
//https://csacademy.com/app/graph_editor/


long long solve(int N, long long K, const std::vector<long long> &A) {
    /* vis.assign(n+1, false);
    g.assign(n+1, vector<int>());
    wg.assign(n + 1, vector<pair<ll,ll>>());
    parent.assign(n+1, -1); */
}

int main() {
    std::ios::sync_with_stdio(false);
    setIO("");
    std::cin.tie(nullptr);
    int N;
    long long K;
    std::cin >> N;
    std::vector<long long> A(N);
    std::cin >> K;
    REP (i, N) {
        std::cin >> A[i];
    }
    auto ans = solve(N, K, A);
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
