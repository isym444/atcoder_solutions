#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <cmath>
#include <bitset>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <deque>
#include <algorithm>
#include <complex>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <cassert>
#include <fstream>
#include <utility>
#include <functional>
#include <time.h>
#include <stack>
#include <array>
#include <list>
#define popcount __builtin_popcount
using namespace std;
typedef long long ll;
typedef pair<int, int> P;

int main()
{
    ll r, x, y;
    cin>>r>>x>>y;
    ll d=x*x+y*y;
    if(d<r*r){
        cout<<2<<endl;
    }else{
        for(ll m=1; m<=1000000; m++){
            if(d<=r*r*m*m){
                cout<<m<<endl;break;
            }
        }
    }
    return 0;
}