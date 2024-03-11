#include<stdio.h>
#include<math.h>
#include<algorithm>
#include<queue>
#include<deque>
#include<stack>
#include<string>
#include<string.h>
#include<vector>
#include<set>
#include<map>
#include<bitset>
#include<stdlib.h>
#include<cassert>
#include<time.h>
#include<bitset>
using namespace std;
const long long mod=998244353;
const long long inf=mod*mod;
const long long d2=(mod+1)/2;
const double EPS=1e-13;
const double PI=acos(-1.0);
int ABS(int a){return max(a,-a);}
long long ABS(long long a){return max(a,-a);}
double ABS(double a){return max(a,-a);}

int main(){
	int a,b;scanf("%d%d",&a,&b);
	if(a>b)swap(a,b);
	if(b==1){
		printf("1\n");return 0;
	}
	if(a==1){
		printf("%d\n",b-2);return 0;
	}
	long long ret=(long long)(a-2)*(b-2);

	printf("%lld\n",ret);
}