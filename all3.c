//892
int surfaceArea(int** grid, int gridSize, int* gridColSize){
    int res = 0;
    int row = gridSize;
    int col = *gridColSize;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (grid[i][j] > 0)
                res += (grid[i][j] * 4 + 2);
            if (i - 1 >= 0 && grid[i - 1][j] > 0)
                res -= grid[i - 1][j] < grid[i][j] ? grid[i - 1][j] : grid[i][j];
            if (i + 1 < row && grid[i + 1][j] > 0)
                res -= grid[i + 1][j] < grid[i][j] ? grid[i + 1][j] : grid[i][j];
            if (j - 1 >= 0 && grid[i][j - 1] > 0)
                res -= grid[i][j - 1] < grid[i][j] ? grid[i][j- 1] : grid[i][j];
            if (j + 1 < col && grid[i][j + 1] > 0)
                res -= grid[i][j + 1] < grid[i][j] ? grid[i][j + 1] : grid[i][j];
        }
    }
    return res;
}

//896
bool isMonotonic(int* A, int ASize){
    if(ASize<=2) return true;
    int flag = 0;
    int minus=0;
    for(int i=1; i<ASize; i++){
        if(A[i-1]==A[i])continue;
        if(A[i-1]!=A[i]) minus = A[i-1]-A[i];
        if(minus*flag<0) return false;
        flag = minus;
    }
    return true;
}

//897
struct TreeNode* inorder(struct TreeNode* root, struct TreeNode* pre) {
    if(root == NULL)
        return pre;
    struct TreeNode* temp = inorder(root->left, root);
    root->left = NULL;
    root->right = inorder(root->right,pre);
    return temp;
}
struct TreeNode* increasingBST(struct TreeNode* root){
    return inorder(root,NULL);
}

//905
int* sortArrayByParity(int* A, int ASize, int* returnSize){
    int startIdx = 0;
    int endIdx = ASize - 1;
    while (startIdx < endIdx) {
        while (startIdx < ASize && A[startIdx] % 2 == 0) {
            startIdx++;
        }
        while (endIdx >= 0 && A[endIdx] % 2 == 1) {
            endIdx--;
        }
        if (startIdx < endIdx) {
            int tmp = A[startIdx];
            A[startIdx] = A[endIdx];
            A[endIdx] = tmp;
        }
    }
    * returnSize = ASize;
    return A;
}

//908
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))
int smallestRangeI(int* A, int ASize, int K){
    int n=ASize;
    int a_min=A[0];
    int a_max=A[0];
    for(int i=1;i<n;++i){
        a_min=min(a_min,A[i]);
        a_max=max(a_max,A[i]);
    }
    return max(a_max-a_min-K-K,0);
}

//910
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))
int cmp(const void *a, const void *b){
    return (*(int*)a - *(int*)b);
}
int smallestRangeII(int* A, int ASize, int K){
    qsort(A,ASize,sizeof(int),cmp);
    int res=A[ASize-1]-A[0];
    for(int i=1;i<ASize;++i){
        int minu=min(A[0]+K, A[i]-K);
        int maxu=max(A[ASize-1]-K,A[i-1]+K);
        res=min(maxu-minu,res);
    }
    return res;
}

//917
char * reverseOnlyLetters(char * S){
    char *P = NULL;
    char *T = NULL;
    char tmp;
    static char Str[101];
    int strLen = 0;
    strLen = strlen(S);
    if (S == NULL)
        return NULL;
    memset(Str, 0, 101* sizeof(char));
    memcpy(Str, S, strLen);
    P = Str;
    T = Str + (strLen - 1);
    while(P < T) {
        if (0 == isalpha(*P)){
            P = P + 1;
            continue;
        }
        if (0 == isalpha(*T)){
            T = T - 1;
            continue;
        }
        tmp = *P;
        *P  = *T;
        *T =  tmp;
        P = P + 1;
        T = T - 1;
    }
    return Str;
}

//922
int* sortArrayByParityII(int* A, int ASize, int* returnSize){
    int even=0,odd=0;
    int *C=(int *)malloc(sizeof(int)*ASize);
    for(int i=0;i<ASize;i++){
        if(A[i]%2==0){
            even=even+1;
            C[(even-1)*2]=A[i];
        }
        else{
            odd=odd+1;
            C[(odd-1)*2+1]=A[i];
        }
    }
    *returnSize=ASize;
    return C;
}

//925
bool isLongPressedName(char * name, char * type){
    int name_num = 0, type_num = 0, i = 0, j = 0;
    for (; name[i] && type[j]; ++i, ++j) {
        for (name_num = 1; name[i] == name[i+1]; ++name_num, ++i);
        for (type_num = 1; type[j] == type[j+1]; ++type_num, ++j);
        if ((name[i] != type[j]) || (type_num < name_num)) return false;
    }
    return (name[i] ? false :true);//如果type字符串结束了，但是name字符串还没结束，即name字符串长于type字符串
}

//933
#define TEST 3000
#define MAX 10000
typedef struct {
    int arr[MAX];
    int header;
    int tail;
    int size;
} RecentCounter;
inline void EnQueue(RecentCounter *q, int val){
    q->arr[q->tail] = val;
    q->tail += 1;
    q->size += 1;
}
inline void DeQueue(RecentCounter *q){
    q->header += 1;
    q->size -= 1;
}
inline int PeekQueue(RecentCounter *q){
    return q->arr[q->header];
}
RecentCounter* recentCounterCreate() {
    RecentCounter *ret = (RecentCounter *)calloc(1, sizeof(RecentCounter));
    if (ret == NULL)
        exit(-1);
    return ret;
}
int recentCounterPing(RecentCounter* obj, int t) {
    EnQueue(obj, t);
    while (t - PeekQueue(obj) > TEST)
        DeQueue(obj);
    return obj->size;
}
void recentCounterFree(RecentCounter* obj) {
    if (obj != NULL)
        free(obj);
}

//938
int rangeSumBST(struct TreeNode* root, int L, int R) {
    if(root == NULL)
        return 0;
    if(root->val > R)
        return rangeSumBST(root->left, L, R);
    else if(root->val < L)
        return rangeSumBST(root->right, L, R);
    else
        return root->val + rangeSumBST(root->left, L, R) + rangeSumBST(root->right, L, R);
}

//941
bool validMountainArray(int* A, int ASize){
    if (ASize < 3)
        return false;
    int indexMax = 0;
    while (indexMax + 1 < ASize && A[indexMax] < A[indexMax + 1])
        indexMax++;
    if (indexMax == 0 || indexMax == ASize - 1)
        return false;
    while (indexMax + 1 < ASize && A[indexMax] > A[indexMax + 1])
        indexMax++;
    return indexMax == ASize - 1;
}

//942
int* diStringMatch(char * S, int* returnSize){
    int n=strlen(S);
    int *r=malloc(sizeof(int)*(n+1));
    int k=0;
    int min=0;
    int max=n;
    for(int i=0;i<n;i++){
        if(S[i]=='I')r[k++]=min++;
        else if(S[i]=='D')r[k++]=max--;
    }
    r[k]=min;
    *returnSize=n+1;
    return r;
}

//944
int minDeletionSize(char ** A, int ASize){
    int n=0;
    int len1=ASize;
    int len2=strlen(A[0]);
    for(int j=0;j<len2;j++){
        for(int i=0;i<len1-1;i++)
            if(A[i][j]>A[i+1][j]){
                n++;
                break;
            }
    }
    return n;
}

//953
bool isAlienSorted(char ** words, int wordsSize, char * order){
    int i,j,sum1,sum2,sum,l,m;
    for(i=1;i<wordsSize;i++){
        sum1=strlen(words[i-1]);
        sum2=strlen(words[i]);
        sum=sum1;
        for(j=0;j<sum;j++){
            if(j==sum2)
                if(sum1>sum2) return false;
            for(l=0;l<26;l++) if(order[l]==words[i-1][j]) break;
            for(m=0;m<26;m++) if(order[m]==words[i][j]) break;
            if(l>m) return false;
            if(l<m) break;
        }
    }
    return true;
}

//961
int repeatedNTimes(int* A, int ASize){
    int nSize = ASize;
    if (nSize == 0)
        return 0;
    int nTag = A[0];
    int nLast = A[1];
    for (int i = 2; i < nSize; ++i){
        int nTmp = A[i];
        if (nTmp == nTag || nTmp == nLast)
            return nTmp;
        else{
            nTag = nLast;
            nLast = nTmp;
        }
    }
    return nLast;
}

//965
bool isUnivalTree(struct TreeNode* root){
    bool left_correct = (root->left == NULL || (root->val == root->left->val && isUnivalTree(root->left)));
    bool right_correct = (root->right == NULL || (root->val == root->right->val && isUnivalTree(root->right)));
    return left_correct && right_correct;
}

//970
bool Isinarr(int *arr,int sum,int m){
    for(int i=0;i<m;++i){
        if(sum==arr[i])
            return false;
    }
    return true;
}
int* powerfulIntegers(int x, int y, int bound, int* returnSize){
    int *arr=(int *)malloc(sizeof(int)*bound);
    int m=0;
    int sum;
    for(int j=0;pow(y,j)<bound&&j<21;++j){
        for(int i=0;pow(x,i)<bound&&i<21;++i){
            sum=pow(x,i)+pow(y,j);
            if(sum<=bound && Isinarr(arr,sum,m))
                arr[m++]=sum;
        }
    }
    *returnSize=m;
    return arr;
}

//976
int *cmp(int *a, int *b) {return *a - *b;}
int largestPerimeter(int* A, int ASize){
    qsort(A, ASize, sizeof(int), cmp);
    for(int i=ASize-1;i>1;i--)
        if(A[i-2]+A[i-1]>A[i])
            return A[i-2]+A[i-1]+A[i];
    return 0;
}

//977
int cmp(const void* a, const void* b){
    return *(int*)a - *(int*)b;
}
int* sortedSquares(int* A, int ASize, int* returnSize){
    int *Array=(int *)malloc(sizeof(int)*ASize);
    for(int i=0;i<ASize;i++)
        Array[i]=A[i]*A[i];
    qsort(Array,ASize,sizeof(Array[0]),cmp);
    *returnSize=ASize;
    return Array;
}

//989
int* addToArrayForm(int* A, int ASize, int K, int* returnSize){
    int bit = 0, k = K;
    while(k){
        bit++;
        k /= 10;
    }
    int len = bit > ASize ? (bit + 1) : (ASize + 1);
    int* ans = (int*)malloc(sizeof(int) * len);
    int i = 0, j = 0;
    for(i = ASize - 1; i >= 0; --i){
        K += A[i];
        ans[j++] = K % 10;
        K /= 10;
    }
    while(K > 0){
        ans[j++] = K % 10;
        K /= 10;
    }
    *returnSize = j;
    j--;
    i = 0;
    while(i < j){
        int t = ans[i];
        ans[i] = ans[j];
        ans[j] = t;
        ++i;
        --j;
    }
    return ans;
}

//993
struct TreeNode* x_f = NULL;
struct TreeNode* y_f = NULL;
bool isCousins(struct TreeNode* root, int x, int y){
    int x_depth = find_depth_x(NULL, root, x);
    int y_depth = find_depth_y(NULL, root, y);
    return (x_f != y_f && x_depth == y_depth);
}
int find_depth_x(struct TreeNode* f_node, struct TreeNode* root, int node_val){
    if (root == NULL)  return -1;
    if (root->val == node_val) {
        x_f = f_node;
        return 0;
    }
    int ret;
    if ((ret = find_depth_x(root, root->left,  node_val)) >= 0) return ret+1;
    if ((ret = find_depth_x(root, root->right, node_val)) >= 0) return ret+1;       
    return -1;
}
int find_depth_y(struct TreeNode* f_node, struct TreeNode* root, int node_val){
    if (root == NULL)  return -1;
    if (root->val == node_val) {
        y_f = f_node;
        return 0;
    }
    int ret;
    if ((ret = find_depth_y(root, root->left,  node_val)) >= 0) return ret+1;
    if ((ret = find_depth_y(root, root->right, node_val)) >= 0) return ret+1;       
    return -1;
}

//994
int orangesRotting(int** grid, int gridRowSize, int *gridColSizes) {
    int good = 0, bad = 0, t = 0;
    for(int i = 0; i < gridRowSize; i++)
        for(int j = 0; j < gridColSizes[0]; j++){
            if(grid[i][j] == 1) good++;//记录好橘子数
            else if(grid[i][j] == 2) bad++; //记录坏橘子数
        }
    if(!good) return 0;
    if(!bad) return -1;
    int new_good = good;
    while(new_good){ //直到好橘子数为零
        t++;//时间加一
        for(int i = 0; i < gridRowSize; i++)
            for(int j = 0; j < gridColSizes[0]; j++)
                if(grid[i][j] == 2){//检查坏橘子四周
                    if(i && grid[i - 1][j] == 1) {
                        grid[i-1][j]=3; 
                        new_good--;
                    }
                    if(j && grid[i][j-1]==1){
                        grid[i][j-1]=3; 
                        new_good--;
                    }
                    if(i != gridRowSize-1 && grid[i+1][j]==1){
                        grid[i+1][j]=3;
                        new_good--;
                    }
                    if(j != gridColSizes[0]-1 && grid[i][j+1]==1){
                        grid[i][j+1]=3;
                        new_good--;
                    }
                }
        for(int i = 0; i < gridRowSize; i++)
            for(int j = 0; j < gridColSizes[0]; j++)
                if(grid[i][j] == 3)
                    grid[i][j] = 2;//重新变为2，方便下一次检查
        if(new_good == good) 
            return -1;//没有新的橘子变坏
        good = new_good;//还剩的好橘子
    }
    return t;
}

//997
int findJudge(int N, int** trust, int trustSize, int* trustColSize){
    int *ret_val = (int *)calloc(N+1, sizeof(int));
    for (int i = 0; i < N+1; ++i)
        ret_val[i] = 0; 

    for (int i = 0; i < trustSize; ++i)
        ++ret_val[trust[i][1]];  //记录被相信的次数

    for (int i = 0; i < trustSize; ++i)
        ret_val[trust[i][0]] = 0;//如果他相信别人，就对被相信的次数清零

    int num = 0;
    int top = -1;
    for (int i = 1; i <= N; ++i) {
        if (ret_val[i] == N-1) {  //查找相信次数为N-1的人
            ++num;
            top = i;//记录下标
        }
    }
    free(ret_val);    
    return (num > 1 ? -1 : top);//如果有大于1个法官，就为-1，否则就为该下标
}

//999
int numRookCaptures(char** board, int boardSize, int* boardColSize){
    int count = 0;
    int x = 0;
    int y = 0;
    for (int i = 0; i < boardSize; i++) {
        for (int j = 0; j < *boardColSize; j++) {
            if (board[i][j] == 'R') {
                x = i;
                y = j;
                break;
            }
        }
    }
    for (int m = x - 1; m >= 0; m--) {
        if (board[m][y] == 'B')
            break;
        else if (board[m][y] == 'p') {
            count++;
            break;
        }
    }
    for (int m = x + 1; m < boardSize; m++) {
        if (board[m][y] == 'B')
            break;
        else if (board[m][y] == 'p') {
            count++;
            break;
        }
    }
    for (int m = y - 1; m >= 0; m--) {
        if (board[x][m] == 'B')
            break;
        else if (board[x][m] == 'p') {
            count++;
            break;
        }
    }
    for (int m = y + 1; m < *boardColSize; m++) {
        if (board[x][m] == 'B')
            break;
        else if (board[x][m] == 'p') {
            count++;
            break;
        }
    }
    return count;
}

//1002
char ** commonChars(char ** A, int ASize, int* returnSize){
    if (A == NULL || ASize <= 0) {
        *returnSize = 0;
        return NULL;
    }
    int arrCnt[ASize][26];
    int minCnt[26];
    int i, j, k, min;
    char ch;
    char **res = (char**)malloc(sizeof(char*) * 100);
    memset(arrCnt, 0, sizeof(int) * ASize * 26);
    for (i = 0; i < 100; i++)
        res[i] = (char*)malloc(sizeof(char) * 2);
    for (i = 0; i < ASize; i++) {
        j = 0;
        while ((ch = A[i][j++]) != '\0')
            arrCnt[i][(int)(ch - 'a')]++;
    }
    for (i = 0; i < 26; i++) {
        min = arrCnt[0][i];
        for (j = 1; j < ASize; j++)
            min = (min > arrCnt[j][i]) ? arrCnt[j][i] : min;
        minCnt[i] = min;
    }

    k = 0;
    for (i = 0; i < 26; i++)
        for (j = 0; j < minCnt[i]; j++) {
            res[k][0] = (char)(i + 'a');
            res[k][1] = '\0';
            k++;
        }
    *returnSize = k;
    return res;
}

//1005
int cmp(const void* a,const void* b){
    return *(int*)a>*(int*)b;
}
int largestSumAfterKNegations(int* A, int ASize, int K){
    qsort(A,ASize,sizeof(int),cmp);
    int sum=0;
    for(int i=0;A[i]<0;i++){
        A[i]=-A[i];
        K--;
        if(K==0){
            for(int i=0;i<ASize;i++)
                sum+=A[i];
            return sum;
        }
    }
    if(K%2==0){
        for(int i=0;i<ASize;i++)
            sum+=A[i];
        return sum;
    }
    qsort(A,ASize,sizeof(int),cmp);
    sum=-A[0];
    for(int i=1;i<ASize;i++)
        sum+=A[i];
    return sum;
}

//1009
int bitwiseComplement(int N){
    if(N == 0)
        return 1;
    int temp = N;
    int j = 0;
    int sum = 0;
    int record;
    while(temp != 0){
        record = temp % 2;
        if(record == 0)
            sum = sum + pow(2,j);
        j++;
        temp = temp/2;       
    }
    return sum;
}

//1010
int numPairsDivisibleBy60(int* time, int timeSize){
    int d[60]={0};
    int temp = 0;
    if (timeSize < 2)
        return 0;
    for(int i = 0;i<timeSize;i++)
        d[time[i]%60]++;
    temp += d[0]*(d[0]-1)/2;
    temp += d[30]*(d[30]-1)/2;
    for(int i = 1;i<30;i++)
        temp += d[i]*d[60-i];
    return temp;
}

//1013
bool canThreePartsEqualSum(int* A, int ASize){
    bool ret = false;
    int sum = 0;
    int i;
    for(i = 0 ; i < ASize; i++)
        sum += A[i];
    // 如果和不能被3整除，返回false
    if((sum % 3) != 0){
        return ret;
    }  
    int s = 0;
    int count = 0;
    int aver = sum / 3;
    for(i = 0; i < ASize; i++){
        s += A[i];
        if(s == aver){
            s = 0;
            count++;
        }
    }
    if(count == 3)
        ret = true;
    return ret;
}

//1018
bool* prefixesDivBy5(int* A, int ASize, int* returnSize){
    if(A == NULL || ASize == 0) {
        *returnSize = 0;
        return NULL;
    }

    bool *ret = (bool *)malloc(sizeof(bool) * ASize);
    *returnSize = ASize;
    memset(ret, 0, sizeof(bool) * ASize);

    unsigned int tmp = 0;
    for (unsigned int i = 0; i < ASize; i++) {
        tmp = (tmp << 1) | (A[i] & 0x1);
        tmp = tmp - (tmp / 5) * 5;
        if(tmp % 5 == 0) {
            ret[i] = true;
        }
    }
    return ret;
}

//1021
char* removeOuterParentheses(char* S){
    int   flag   = 0;
    int   flag_l = 0;
    char* n      = malloc(strlen(S) + 1);
    int   idx    = 0;
    memset(n, 0, strlen(S) + 1);
    while(*S){
        if(*S == '(')
            flag++;
        if(*S == ')')
            flag--;
        if((flag_l == 0) && (flag == 1))
            S++;
        else if((flag_l == 1) && (flag == 0))
            S++;
        else{
            n[idx++] = *S;
            S++;
        }
        flag_l = flag;
    }
    return n;
}

//1022
int _sumRootToLeaf(struct TreeNode *root, int num) {
    int sum = 0; 
    num = (num << 1) + root->val;
    if (root->left == NULL && root->right == NULL) return num; 
    if (root->left) sum += _sumRootToLeaf(root->left, num);
    if (root->right) sum += _sumRootToLeaf(root->right, num);
    return sum; 
}

int sumRootToLeaf(struct TreeNode *root) {
    return root ? _sumRootToLeaf(root, 0) : 0;
}

//1025
bool divisorGame(int N){
  return !(N%2);
}

//1029
#define min(a,b) (((a) < (b)) ? (a) : (b))
int twoCitySchedCost(int** costs, int costsSize, int* costsColSize){
    int n=costsSize;
    *costsColSize=n+1;
    int **dp=(int**)calloc((n+1),sizeof(int*));
    for(int i=0;i<=n;i++){
        dp[i]=(int*)calloc((n+1),sizeof(int));
    }
    for(int i=0;i<=n;i++){
        for(int j=i+1;j<=n;j++) 
            dp[i][j]=1e9;
    }
    for(int i=1;i<=n;i++){
        dp[i][0]=dp[i-1][0]+costs[i-1][1];
        for(int j=1;j<=i;j++){
            dp[i][j]=min(dp[i-1][j]+costs[i-1][1],dp[i-1][j-1]+costs[i-1][0]);
        }
    }
    return dp[n][n/2];
}

//1033
int* numMovesStones(int a, int b, int c, int* returnSize){
    *returnSize = 2;
    int *res = (int*)calloc(*returnSize, sizeof(int));
    int t;
    if(a > b){
        t = a;
        a = b;
        b = t;
    }
    if(a > c){
        t = a;
        a = c;
        c = t;
    }
    if(b > c){
        t = b;
        b = c;
        c = t;
    }
    if (a + 1 == b && b + 1 == c)
        return res;
    else if (a + 1 == b || b + 1 == c){
        res[0] = 1;
        if (a + 1 != b)
            res[1] = b - a - 1;
        else
            res[1] = c - b - 1;
    } 
    else if (a + 2 == b || b + 2 == c){
        res[0] = 1;
        res[1] = (b - a - 1) + (c - b - 1);
    }
    else{
        res[0] = 2;
        res[1] = (b - a - 1) + (c - b - 1);
    }
    return res;
}

//1037
bool isBoomerang(int** points, int pointsSize, int* pointsColSize){
    int x1 = points[0][0] - points[1][0];
    int y1 = points[0][1] - points[1][1];
    int x2 = points[0][0] - points[2][0];
    int y2 = points[0][1] - points[2][1];
    return x1 * y2 != x2 * y1;
}

//1042
int* gardenNoAdj(int N, int** paths, int pathsSize, int* pathsColSize, int* returnSize){
    int *result = (int *)calloc(N , sizeof(int));
    int i, j, r, m;
    int (*map)[3] = calloc(3 * N, sizeof(int)); //标号i的花园对应的临近花园
    int *order = calloc(N, sizeof(int)); //标记下一个临近花园插入数组map[i]的位置

    result[0] = 1; //先给1号花园种1号花
    if(N == 1){    //只有一个花园
        *returnSize = N;
        return result;
    }
    for(i = 0; i < pathsSize; i++){ //建立map[i]
        map[paths[i][0] - 1][order[paths[i][0] - 1]++] = paths[i][1];
        map[paths[i][1] - 1][order[paths[i][1] - 1]++] = paths[i][0];
    }
    for(i = 0; i < N; i++){ //找到与临近花园花种不同的花
        if(map[i][0] == 0){ // map[i][0]等于0代表i号花园没有临近的花园，直接种1号花
            result[i] = 1;
            continue;
        }
        for(r = 1; r < 5; r++){ //从1-4尝试各个花种直到与其他临近花园的花种都不同
            for(j = 0, m = 0; j < 3; j++){
                if(map[i][j] == 0)
                    m++;
                else if(result[map[i][j] - 1] != r)
                    m++;
            }
            if(m == 3){
                result[i] = r;
                break;
            }
        }
    }
    *returnSize = N;
    return result;
}

//1043
#define max(a,b) (((a) > (b)) ? (a) : (b))
int maxSumAfterPartitioning(int* A, int ASize, int K){
    int *dp = (int*)calloc(ASize+1,sizeof(int));
    for(int i=1;i<=ASize;++i){
        int maxtemp=A[i-1];
        for(int j=1;j<=K;++j){
            if(i-j>=0){
                maxtemp = max(maxtemp,A[i-j]);//标准就是A[i-1] 即求A[i-k]~A[i-1]中的最大值
                dp[i]=max(dp[i],dp[i-j]+j*maxtemp);
            }
        }
    }
    return dp[ASize];
}

//1046
int lastStoneWeight(int* stones, int stonesSize){
    if(stones==NULL)
        return 0;
    int i,j;
    int temp,max;
    while(stonesSize>1){
        //两次选择排序
        for(j=1;j<=2;j++){
            max=stones[stonesSize-j];
            for(i=stonesSize-j;i>=0;i--){
                if(max<stones[i]){
                    max=stones[i];//关键语句：更新最大值
                    temp=stones[i];
                    stones[i]=stones[stonesSize-j];
                    stones[stonesSize-j]=temp;
                }
            }
        }
        stones[stonesSize-2]=stones[stonesSize-1]-stones[stonesSize-2];
        stonesSize--;
    }
    return stones[0];
}

//1047
char * removeDuplicates(char * S){
    int write=0;
    int read=0;
    char top, ch;    
    while(ch = S[read++]){
        if(0 == write) // empty string
            top = 0;
        else
            top = S[write-1];
        if(ch == top) //delete top character
           write--;
        else // write current character to top
            S[write++] = ch;                   
    }
    S[write] = 0;
    return S;
}

//1048
#define max(a,b) (((a) > (b)) ? (a) : (b))
void sort(char **nums,int low,int high){
    if(low>=high) return;
    int i = low,j = high;
    char* temp = nums[i];
    while(i<j){
        while(i<j&&strlen(nums[j])>strlen(temp))    --j;
        nums[i] = nums[j];
        while(i<j&&strlen(nums[i])<=strlen(temp))   ++i;
        nums[j] = nums[i];
    }
    nums[i] = temp;
    sort(nums,low,i-1);
    sort(nums,i+1,high);
}
int isFor(char *a, char *b){
    if(strlen(b)-strlen(a)==1){
        int i=0,j=0;
        while(i<strlen(a)&&j<strlen(b)){
            if(a[i]==b[j]) i++;
            j++;
        }
        if(i==strlen(a)) return 1;
    }
    return 0;
}
int longestStrChain(char ** words, int wordsSize){
    if(wordsSize<2) return wordsSize;
    int dp[wordsSize+2];
    for(int i=0;i!=wordsSize+2;i++)
        dp[i]=1;
    int res=1;
    sort(words,0,wordsSize-1);
    for(int i=0;i<wordsSize;i++){
        for(int j=i-1;j>=0;j--){
            if(isFor(words[j],words[i])==1){
                dp[i]=max(dp[i],dp[j]+1);
            }
        }
        res=max(res,dp[i]);
    }
    return res;
}

//1051
int heightChecker(int* heights, int heightsSize){
    int i = 0, j = 0, sum = 0, tmp = 0;
    int list[101] = {0};
    for(i=0;i<heightsSize;i++)
        list[heights[i]]++;
    for(i=1;i<101;i++)
        while(list[i]>0){
            if(heights[j] != i) sum++;
            list[i]--;
            j++;
        }
    return sum;
}

//1071
int TOP = 0;
char * gcdOfStrings(char * str1, char * str2){
    int len1 = strlen(str1);
    int len2 = strlen(str2);
    char *order_str  = (len1 > len2 ? str2 : str1);
    int   o_len      = (len1 > len2 ? len2 : len1);
    char *tmp_str    = (char *) malloc(1024);
    strcpy(tmp_str, order_str);
    for (int tmp_len = o_len; tmp_len > 0; --tmp_len, tmp_str[tmp_len] = '\0') {
        if (!(len1 % tmp_len) && !(len2 % tmp_len)) { //字符串长度必须整除
            int tmp_1 = find_str(str1, tmp_str);
            if (tmp_1 < tmp_len || TOP < len1) continue;  //判断是否同时都比较到字符串尾部
            int tmp_2 = find_str(str2, tmp_str);
            if (tmp_2 < tmp_len || TOP < len2) continue;
            return tmp_str;
        }
    }
    return tmp_str;
}

int find_str(char *master, char *order){
    TOP = 0;
    int j;
    while (master[TOP]) {
        for (j = 0; order[j]; ++j) {//重复比较
            if (order[j] != master[TOP]) return j;
            ++TOP;
        }
    }
    return j;
}

//1078
int find(char *text, int i){
    while (text[i] != '\0' && text[i] != ' ') ++i;//找到一个单词的结尾出
    return i;
}
char ** findOcurrences(char * text, char * first, char * second, int* returnSize){
    int flage = 0;
    int return_size = 0;
    int str_len = strlen(text) + 1;
    int i = 0;
    int j = 0;
    int sum = 0;
    int row = 64;
    int cloumn = 64;
    char **a = (char **)malloc(sizeof(char *) * row);
    for (i = 0; i < row; ++i) {
        a[i] = (char *)malloc(sizeof(char) * cloumn);
    }
    for (i = 0; i < str_len; ++i) {
        j = find(text, i);
        text[j] = '\0';
        if (flage == 2) {
            strcpy(a[sum++], text + i);
            if (!strcmp(text+i, first)) flage = 1;
            else flage = 0;
        }else if (!strcmp(text+i, first)) {
            flage = 1;
        }else if ((!strcmp(text+i, second)) && (flage == 1)){
            flage = 2;
        }else flage = 0;
        i = j;
    }
    *returnSize = sum;
    return a;
}

//1122
int* relativeSortArray(int* arr1, int arr1Size, int* arr2, int arr2Size, int* returnSize){
    int mark[1001]={0};
    for(int i=0;i<arr1Size;i++)
        mark[arr1[i]]++;
    int j=0;
    for(int i=0;i<arr2Size;i++)
        if(mark[arr2[i]]>0)
            for(int k=mark[arr2[i]];k>0;k--){
                arr1[j]=arr2[i];
                j++;
                mark[arr2[i]]--;
            }
    for(int i=0;i<1001;i++)
        if(mark[i]>0)
            for(int k=mark[i];k>0;k--){
                arr1[j]=i;
                j++;
                mark[i]--;
            }
    *returnSize=arr1Size;
    return arr1;
}

//1128
int pairs[9][9] = {0};
int calcTotal(int n){
    return (n * (n - 1)) >> 1; 
}
int numEquivDominoPairs(int** dominoes, int dominoesSize, int* dominoesColSize){
    int cnt = 0;
    memset(pairs, 0, sizeof(pairs));
    if (dominoesSize > 40000) {
        return 0;
    }
    for (int i = 0; i < dominoesSize; i++) {
        pairs[dominoes[i][0] - 1][dominoes[i][1] - 1]++; 
    }
    for (int i = 0; i <= 8; i++) {
        for (int j = 0; j < i; j++) {
            int num = pairs[i][j] + pairs[j][i];
            if (num) {
                cnt += calcTotal(num); 
            }
        }
    }
    for (int i = 0; i <= 8; i++) {
        int num = pairs[i][i];
        if (num) {
            cnt += calcTotal(num); 
        }
    }
    return cnt;
}

//1137
int tribonacci(int n){
    int dp[38];
    int i;
    dp[0] = 0;
    dp[1] = 1;
    dp[2] = 1;
    for (i = 3; i <= n; i++) {
        dp[i] = dp[i - 3] + dp[i - 2] + dp[i - 1];
    }
    return dp[n];
}

//1154
int monthday[2][12] = {
    {31,28,31,30,31,30,31,31,30,31,30,31},
    {31,29,31,30,31,30,31,31,30,31,30,31}
}; 
int dayOfYear(char * date){
    char *y, *m, *d;
    y = date;
    m = &date[5];
    d = &date[8];
    date[4] = '\0';
    date[7] = '\0';
    int iy, im, id;
    iy = atoi(y);
    im = atoi(m);
    id = atoi(d);
    int isNormalYear = 0;
    if ((iy % 100 != 0 && iy % 4 == 0) || iy % 400 == 0) {
        isNormalYear = 1;
    }
    int ret = 0;
    for (int i = 0; i < im - 1; i++) {
        ret += monthday[isNormalYear][i];
    }
    return ret + id;
}

//1156
#define max(a, b)  a > b ? a : b;
int maxRepOpt1(char * text){
    if(text == 0) {
        return 0;
    }
    int slen = strlen(text);
    int num[26] = {0};
    int *left = malloc(slen * sizeof(int));
    int *right = malloc(slen * sizeof(int));
    int res = 0;
    for(int i = 0; i < slen; i++){
        left[i] = 1;
        right[i] = 1;
    }
    for(int i = 0; i < slen; i++){
        num[text[i] - 'a']++;
    }
    int c = 1;
    for(int i = 1; i < slen; i++) {
        if(text[i] == text[i-1]) {
            c++;
        }
        else{
            c = 1;
        }
        left[i] = c;
        res = res > left[i] ? res : left[i];
    }
    c = 1;
    for(int i = slen-2; i >= 0; i--) {
        if(text[i] == text[i + 1]) {
            c++;
            right[i] = c;
        }
        else{
            c = 1;
        }
        res = res > right[i] ? res : right[i];
    }
    for(int i = 1;i < slen-1; i++){
        if(num[text[i-1] - 'a'] > left[i-1]) {
            res = res > left[i-1] + 1 ? res : left[i-1] + 1;
        }
        if(num[text[i+1]- 'a'] > right[i+1]){
            res = res > right[i + 1] + 1 ? res : right[i + 1] + 1; 
        }
        if(text[i-1] == text[i+1] && text[i-1] != text[i]){
            if(num[text[i-1]-'a'] > (left[i-1] + right[i+1])){
                res = res > left[i-1] + right[i+1] + 1 ? res : left[i-1] + right[i+1] + 1;
            }else{
                res = res > left[i-1] + right[i+1] ? res : left[i-1] + right[i+1];
            }
        }
    }
    return res;
}

//1160
int countCharacters(char ** words, int wordsSize, char * chars){
    char mychars[100] = {0};
    int num = 0;
    for (int i = 0; i < wordsSize; i++) {
        memcpy(mychars, chars, strlen(chars));
        int j = 0;
        char *first = NULL;
        while (words[i][j] != '\0') {
            if ((first = strchr(mychars, words[i][j])) != NULL) {
                int weizhi = first - mychars;
                mychars[weizhi] = '_';
                j++;
            } else
                break;
        }
        if (j == strlen(words[i]))
            num += j;
    }
    return num;
}

//1170
int fs(char* str){
    int i = 0;
    char min = 'z';
    int minCnt = 0;
    for (i = 0; i < strlen(str); i++){
        if (str[i] < min){
            min = str[i];
            minCnt = 1;
        }else if (str[i] == min) {
            minCnt++;
        }
    }
    return minCnt;
}

int* numSmallerByFrequency(char ** queries, int queriesSize, char ** words, int wordsSize, int* returnSize){        
    int counter[12] = {0};
    int* answer = malloc(queriesSize*sizeof(int));
    memset(answer, 0, queriesSize*sizeof(int));
    // 统计
    for (int i = 0; i < wordsSize; i++)
        counter[fs(words[i])]++;
    // 累和
    for (int i = 9; i >= 0; i--)
        counter[i] += counter[i + 1];
    // 拿值
    for (int i = 0; i < queriesSize; i++)
        answer[i] = counter[fs(queries[i]) + 1];
    *returnSize = queriesSize;
    return answer;
}

//1175
bool isPrime(int num){
    if(num<2)
        return false;
    int N=sqrt(num);
    for(int i=2;i<=N;++i){
        if(num%i==0)
            return false;
    }
    return true;
}
int numPrimeArrangements(int n) {
    int count=0;
    for(int i=1;i<=n;++i){
        if(isPrime(i))
            ++count;
    }
    long long res=1;
    int temp=n-count;
    int div=pow(10,9)+7;
    while(count>0){
        res*=count--;
        if(res>=div)
            res%=div;
    }
    while(temp>0){
        res*=temp--;
        if(res>=div)
            res%=div;
    }
    return res;
}

//1184
int distanceBetweenBusStops(int* distance, int distanceSize, int start, int destination){
    if ((distanceSize == 0) || (start == destination))
        return 0;
    int maxlen = 0;
    int leninorder = 0;    
    if (start > destination) {
        int temp;
        temp = start;
        start = destination;
        destination = temp;
    }    
    for (int loop  = 0; loop < distanceSize; loop++) {
        if ((loop >= start) && (loop < destination))
            leninorder += distance[loop];
        maxlen += distance[loop];  
    }
    return fmin(maxlen - leninorder, leninorder);
}

//1185
char * dayOfTheWeek(int day, int month, int year){
    char* week[7]={"Friday","Saturday","Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"};
    int dc,yc;
    int rc;
    yc=year-1971;
    if(yc%4==2||yc%4==3)
        rc=yc/4+1;
    else
        rc=yc/4;
    dc=yc*365+rc;
    int i=0;
    if(year%4==0)
        i=1;
    switch(month){
        case 1:dc=dc+day;break;
        case 2:dc=dc+day+31;break;
        case 3:dc=dc+day+59+i;break;
        case 4:dc=dc+day+90+i;break;
        case 5:dc=dc+day+120+i;break;
        case 6:dc=dc+day+151+i;break;
        case 7:dc=dc+day+181+i;break;
        case 8:dc=dc+day+212+i;break;
        case 9:dc=dc+day+243+i;break;
        case 10:dc=dc+day+273+i;break;
        case 11:dc=dc+day+304+i;break;
        case 12:dc=dc+day+334+i;break;
    }
    return week[(dc-1)%7];
}

//1189
#define MIN(a, b) ((a) < (b) ? (a) : (b))
int maxNumberOfBalloons(char * text){
    int cnt[26] = {};
    int len = strlen(text);
    for (int i = 0; i < len; i++)
        cnt[text[i] - 'a']++;
    int min = cnt['a' - 'a'];
    min = MIN(min, cnt['b' - 'a']);
    min = MIN(min, cnt['l' - 'a'] / 2);
    min = MIN(min, cnt['o' - 'a'] / 2);
    min = MIN(min, cnt['n' - 'a']);
    return min;
}

//1207
bool uniqueOccurrences(int* arr, int arrSize){
    int hash_table[2000];
    int hash_res_table[2000];
    memset(hash_table, 0, sizeof(int)*2000);
    memset(hash_res_table, 0, sizeof(int)*2000);
    int i = 0;
    while(i < arrSize){
        if(arr[i] < 0)
            hash_table[(arr[i] * -1)]++;
        else
            hash_table[(arr[i] + 1000)]++;
        i++;
    }
    i = 0;
    while(i < 2000){
        if(hash_table[i] > 0)
            hash_res_table[hash_table[i]]++;
        if(hash_res_table[hash_table[i]] > 1)
            return false;
        i++;
    } 
    return true;
}

//1217
int minCostToMoveChips(int* chips, int chipsSize){
    int i = 0;
    int countA = 0,countB = 0;
    for( i = 0; i < chipsSize;i++){
        if(chips[i] & 0x01)
            countA++;
        else
            countB++;
    }
    return (countA > countB) ? countB:countA; 
}

//1221
int balancedStringSplit(char * s){
    int r=0, l=0, sum=0;
    while(*s){
        if(*s=='R') r+=1;
        else if(*s=='L') l+=1;
        if(r==l) sum+=1;
        s+=1;
    }
    return sum;
}

//1232
bool checkStraightLine(int** coordinates, int coordinatesSize, int* coordinatesColSize){
    double x=coordinates[1][0]-coordinates[0][0],y=coordinates[1][1]-coordinates[0][1];
    double k;
    int t=0;
    if(y==0)
        t=1;
    else
        k=x/y;
    for(int i=1;i<coordinatesSize-1;i++){
        y=coordinates[i+1][1]-coordinates[i][1];
        if(t&&y)
            return false;
        x=coordinates[i+1][0]-coordinates[i][0];
        if(!t&&k!=x/y)
            return false;
    }
    return true;
}

//1234
#define min(a,b) (((a) < (b)) ? (a) : (b))
int balancedString(char * s){
    int *count = (int*)calloc(26,sizeof(int));//QWER的个数数组
    int len=strlen(s);
    for(int i=0;i!=strlen(s);i++){
        count[s[i]-'A']++;
    }
    int left = 0, right = 0;
    int res = len;
    int average = len / 4;

    while(right<len){
        count[s[right]-'A']--;
        while(left < len 
            && count['Q'-'A'] <= average 
            && count['W'-'A'] <= average 
            && count['E'-'A'] <= average 
            && count['R'-'A'] <= average ){
            res = min(res, right - left + 1);
            count[s[left]-'A']++;
            left++;
        }
        right++;
    }
    return res;
}

//1249
char * minRemoveToMakeValid(char * s){
    int j = 0;
    int k = 0;
    int len = strlen(s);
    int *cz = malloc(len * sizeof(int));
    if (cz == NULL){
        return NULL;
    }
    memset(cz, -1, len * sizeof(int));
    int *cy = malloc(len * sizeof(int));
    if (cy == NULL)
        return NULL;
    memset(cy, -1, len * sizeof(int));
    char *res = malloc((len + 1) * sizeof(char));
    if (res == NULL)
        return NULL;
    for (int i = 0; i < len; i++) {
        if (s[i] == '(') {
            cz[j] = i;
            j++;
        }
        if (s[i] == ')') {
            if (j > 0) {
                j--;
                cz[j] = -1;
            } else {
                cy[k] = i;
                k++;
            }
        }
    }
    int z = 0;
    int v = 0;
    int w = 0;
    for (int i = 0; i < len; i++) {
        if (i == cz[z]) {
            z++;
            continue;
        } else if (i == cy[v]) {
            v++;
            continue;
        }else {
            res[w] = s[i];
            w++;
        }
    }
    res[w] = 0;
    return res;
}

//1252
int oddCells(int n, int m, int** indices, int indicesSize, int* indicesColSize){
    int res = 0;
    int i, j;
    int *a = (int*)malloc(n * sizeof(int));
    memset(a, 0, n*sizeof(int));
    int *b = (int*)malloc(m * sizeof(int));
    memset(b, 0, m*sizeof(int));
    for (i = 0; i < indicesSize; i++){
        a[indices[i][0]] = (a[indices[i][0]]+1)%2;
        b[indices[i][1]] = (b[indices[i][1]]+1)%2;
    }
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            if (a[i] != b[j])
                res++;
    return res;
}

//LCP 1
int game(int* guess, int guessSize, int* answer, int answerSize){
    return (guess[0]==answer[0])+(guess[1]==answer[1])+(guess[2]==answer[2]);
}

//LCP 2
int arr[2];
int* fraction(int* cont, int contSize, int* returnSize){
    int i = contSize-1;
    int denominator = 0;
    int molecule = 0;
    for(;i>=0;i--){
       if(i==contSize-1){
            arr[1] = 1;
            arr[0] = *(cont+i);
        }else{
            denominator = arr[0];
            molecule = arr[1]+(*(cont+i)*arr[0]);
            arr[1] = denominator;
            arr[0] = molecule;
        }
    }
    *returnSize = 2;
    return arr;
}
