//001
typedef struct hash_node {
    int id;            /* we'll use this field as the key */
    int index;
    UT_hash_handle hh; /* makes this structure hashable */
} hash_node;

int* twoSum(int* nums, int numsSize, int target, int* returnSize){
    int *two_nums = (int *)malloc(sizeof(int)*2);
    hash_node *hash_table = NULL, *hash_item1 = NULL, *hash_item2 = NULL;
    for (int i = 0; i < numsSize; i++) {
        // 查找哈希表中是否存在满足和为target的另一个值,若存在直接返回
        int other_id = target - nums[i];
        HASH_FIND_INT(hash_table, &other_id, hash_item1);
        if (hash_item1) {
            two_nums[0] = hash_item1->index;
            two_nums[1] = i;
            *returnSize = 2;
            return two_nums;
        }
        // 将本次遍历的值放入哈希表,value为数组下标,key为对应数值
        hash_item2 = (hash_node *)malloc(sizeof(hash_node));
        hash_item2->id = nums[i];
        hash_item2->index = i;
        HASH_ADD_INT(hash_table, id, hash_item2);
    }
    return two_nums;
}

//002
struct ListNode* addTwoNumbers(struct ListNode* l1, struct ListNode* l2) {
    int c = 0;
    struct ListNode *head, *cur, *next;
    head = (struct ListNode *)malloc(sizeof(struct ListNode));
    head->next = NULL;
    cur = head;
    while(l1!=NULL || l2!=NULL || c) {
        next=(struct ListNode *)malloc(sizeof(struct ListNode));
        next->next=NULL;
        cur->next=next;
        cur=next;
        l1!=NULL?(c+=l1->val,l1=l1->next):(c+=0);
        l2!=NULL?(c+=l2->val,l2=l2->next):(c+=0);
        cur->val=c%10;
        c=c/10;
    }
    struct ListNode *del = head;
    head=head->next;
    free(del);
    return head;
}

//003
int lengthOfLongestSubstring(char * s){
    int prior = 0; //上次状态下最长子串的长度
    int dict[256] = {0}; //映射ASCII码
    int i;
    for (int left = 0, right = 1; *s != '\0'; right++){
        i = *s-0; //字符转换为整数
        if(dict[i] > left)    
            left = dict[i];
        dict[i] = right;
        prior = (prior>right-left)?prior:right-left; //right的值比对应的数组下标大1
        s++;
    }
    return prior;
}

//004
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))
double findMedianSortedArrays(int* nums1, int nums1Size, int* nums2, int nums2Size){
    if (nums1Size>nums2Size)
        return findMedianSortedArrays(nums2, nums2Size, nums1, nums1Size);
    // Ci 为第i个数组的割,比如C1为2时表示第1个数组只有2个元素。LMaxi为第i个数组割后的左元素。RMini为第i个数组割后的右元素。
    int LMax1, LMax2, RMin1, RMin2, c1, c2, lo = 0, hi = 2 * nums1Size;//我们目前是虚拟加了'#'所以数组1是2*n长度
    while (lo <= hi){   //二分
        c1 = (lo + hi) / 2;  //c1是二分的结果
        c2 = nums1Size + nums2Size - c1;
        LMax1 = (c1 == 0) ? INT_MIN : nums1[(c1 - 1) / 2];
        RMin1 = (c1 == 2 * nums1Size) ? INT_MAX : nums1[c1 / 2];
        LMax2 = (c2 == 0) ? INT_MIN : nums2[(c2 - 1) / 2];
        RMin2 = (c2 == 2 * nums2Size) ? INT_MAX : nums2[c2 / 2];
        if (LMax1 > RMin2) hi = c1 - 1;
        else if (LMax2 > RMin1) lo = c1 + 1;
        else break;
    }
    return (max(LMax1, LMax2) + min(RMin1, RMin2)) / 2.0;
}

//005
#define max(a,b) (((a) > (b)) ? (a) : (b))
int expand(char * s, int l, int r){
    while(l>=0 && r<strlen(s) && s[l]==s[r]){
        l--;
        r++;
    }
    return r-l-1;
}
char * longestPalindrome(char * s){
    if (strlen(s)==0) return "";
    int umax = 1, umid = 0, size=strlen(s);
    for (int i=0; i<size-1; i++){
        int temp = max(expand(s,i,i+1), expand(s,i,i));
        if ((temp>umax) && temp>1){
            umid=i;
            umax=temp;
        }
    }
    int l=umid-(umax-1)/2;
    int r=umid+umax/2+1;
    char *ans = (char*)malloc(sizeof(char)*(r-l+1));
    memset(ans,0,r-l+1);
    strncpy(ans,s+l,r-l);
    return ans;
}

//007
int reverse(int x){
    int ans = 0;
    while(x!=0){
        int pop=x%10;
        if(ans>INT_MAX /10||(ans==INT_MAX/10 && pop>7)) return 0;
        if(ans<INT_MIN/10||(ans==INT_MIN/10 && pop<-8)) return 0;
        ans = ans*10+pop;
        x/=10;
    }
    return ans;
}

//009
bool isPalindrome(int x){
    int y = x ;  //用y记住原先值
    long res=0;  //直接定义为long型，用于存储翻转后的数据，因为int型取值范围内倒转可能会溢出
    while(x > 0){
         res = res*10 + x%10;   
         x = x/10;   
    }
    if(res == y) return true;
    return false;
}

//013
int mp(char c){
    switch(c){
        case 'I': return 1;
        case 'V': return 5;
        case 'X': return 10;
        case 'L': return 50;
        case 'C': return 100;
        case 'D': return 500;
        case 'M': return 1000;
        default: return 0;
    }
}

int romanToInt(char * s){
    int ans=0;
    for(int i=0;i<strlen(s)-1;i++){
        if(mp(s[i])<mp(s[i+1]))
            ans-=mp(s[i]);
        else
            ans+=mp(s[i]);
    }
    ans+=mp(s[strlen(s)-1]);
    return ans;
}

//014
int min(int x,int y,int z){
    int min=x;
    if (y<min) min=y;
    if (z<min) min=z;
    return min;
}
char * longestCommonPrefix(char ** strs, int strsSize){
    strsSize=0;
    char *str0=strs[0]; int len0=strlen(str0);
    char *str1=strs[1]; int len1=strlen(str1);
    char *str2=strs[2]; int len2=strlen(str2);
    int size=min(len0,len1,len2);
    for(int i=0;i!=size;i++){
        if(str0[i]==str1[i] && str0[i]==str2[i])
            strsSize++;
    }
    if(strsSize!=0){
        char *ans=(char*)malloc(sizeof(char)*strsSize);
        strncpy(ans,str0,strsSize);
        return ans;
    }
    return "";
}

//020
bool isValid(char * s){
    if (s == NULL || s[0] == '\0') return true;
    char *stack = (char*)malloc(strlen(s)+1); int top =0;
    for (int i = 0; s[i]; ++i) {
        if (s[i] == '(' || s[i] == '[' || s[i] == '{') stack[top++] = s[i];
        else {
            if ((--top) < 0)                      return false;//先减减，让top指向栈顶元素
            if (s[i] == ')' && stack[top] != '(') return false;
            if (s[i] == ']' && stack[top] != '[') return false;
            if (s[i] == '}' && stack[top] != '{') return false;
        }
    }
    return (!top);//防止“【”这种类似情况
}

//021
struct ListNode* mergeTwoLists(struct ListNode* l1, struct ListNode* l2){
    if(l1 == NULL) return l2;
    if(l2 == NULL) return l1;
    struct ListNode *head = (struct ListNode*)malloc(sizeof(struct ListNode));
    struct ListNode *p = head;
    while (l1 && l2){
        if (l1 -> val < l2 -> val){
            head -> next = l1;
            head = l1;
            l1 = l1 -> next; 
        }
        else{
            head -> next = l2;
            head = l2;
            l2 = l2 -> next;
        }
    }
    head -> next = l1 ? l1 : l2;
    return p -> next;
}

//026
int removeDuplicates(int* nums, int numsSize){
    if(numsSize==0) return 0;
    int i=0;
    for(int j=1;j<numsSize;j++)
        if(nums[j]!=nums[i]){
            i++;
            nums[i]=nums[j];
        }
    return i+1;
}

//027
int removeElement(int* nums, int numsSize, int val){
    int i = 0;
    for (int j = 0; j < numsSize; j++) {
        if (nums[j] != val) {
            nums[i] = nums[j];
            i++;
        }
    }
    return i;
}


//028 +
int strStr(char * haystack, char * needle){
    int len2 = strlen(needle);
    if (!len2) return 0;
    int len1 = strlen(haystack), j;
    for (int i = 0; i <= len1 - len2; i++)
        if (haystack[i] == needle[0] && haystack[i + len2 - 1] == needle[len2 - 1]){
            for (j = 1; j < len2; j++)
                if (haystack[i + j] != needle[j])
                    break;
            if (j == len2)
                return i;
        }
    return -1;
}

//035
int searchInsert(int* nums, int numsSize, int target){
    int l=0, r=numsSize-1, m;
    while(l<=r){
        m=l+((r-l)>>1);
        if(nums[m] == target) return m;
        if(target < nums[m]) r = m-1;
        if(target > nums[m]) l = m+1;
    }
    if(target <nums[m]) return m;
    return m+1;
}

//053
#define max(a,b) (((a)>(b))?(a):(b))
int maxSubArray(int* nums, int numsSize){
    int ans=nums[0];
    int sum=0;
    for(int i=0;i!=numsSize;i++){
        if(sum>0) sum+=nums[i];
        else sum=nums[i];
        ans=max(ans,sum);
    }
    return ans;
}

//058
int lengthOfLastWord(char * s){
    int len=strlen(s);
    if(len==0) return 0;
    int i=len-1;
    while(i>=0&&s[i]==' ') i--;
    len=i;
    while(i>=0&&s[i]!=' ') i--;
    return len-i;
}

//066
int* plusOne(int* digits, int digitsSize, int* returnSize){
    *returnSize=digitsSize;
    for(int i=digitsSize-1;i>=0;i--){
        digits[i]++;
        digits[i]%=10;
        if(digits[i]!=0) return digits;
    }
    *returnSize=digitsSize+1;
    digits=(int*)realloc(digits,*returnSize*(sizeof(int)));
    digits[0]=1;
    digits[digitsSize]=0;
    return digits;
}


//067
void Reverse(char *s,int n){ 
    for(int i=0,j=n-1;i<j;i++,j--){ 
        char c=s[i]; 
        s[i]=s[j]; 
        s[j]=c; 
    } 
}

char * addBinary(char * a, char * b){
    int c=0,i=strlen(a)-1,j=strlen(b)-1,k=0;
    char *s=(char*)calloc(i+j+3,sizeof(char));
    while(i >= 0 || j >= 0 || c == 1){
        c += i >= 0 ? a[i--] - '0' : 0;
        c += j >= 0 ? b[j--] - '0' : 0;
        s[k]=(c & 1) + '0';
        k++;
        c >>= 1;
    }
    Reverse(s,strlen(s));
    return s;
}

//069 牛顿迭代法
int mySqrt(int x){
    if(x<2)
        return x;
    double t = x;
    double x0 = x;
    x0 = x0/2 + t/(2*x0);
    while(x0*x0 - t > 0.00001)
        x0 = x0/2 + t/(2*x0);
    return (int)x0;
}

//070
int climbStairs(int n){
    if(n<=2)
        return n;
    int i1 = 1, i2 = 2;
    for(int i=3;i<=n;i++){
        int temp = i1+i2;
        i1 = i2;
        i2 = temp;
    }
    return i2;
}

//083
struct ListNode* deleteDuplicates(struct ListNode* head){
    struct ListNode *p, *next;
    p = next = head;
    while (p!=NULL) {
        while(next!=NULL && next->val==p->val) {
            next=next->next;
        }
        p->next=next;
        p=next;
    }
    return head;
}

//088
void merge(int* nums1, int nums1Size, int m, int* nums2, int nums2Size, int n){
    while(m&&n)
        nums1[m+n]=nums1[m-1]>nums2[n-1]?nums1[--m]:nums2[--n];
    while(n)nums1[n]=nums2[--n];
}

//091
int numDecodings(char * s) {
    if (s[0]=='0'||s[0]==0) return 0;
    int pre=1,curr=1;//dp[-1]=dp[0]=1
    for(int i=1;s[i]!=0;i++){
        int tmp=curr;
        if(s[i]=='0')
            if(s[i-1]=='1'||s[i-1]=='2') curr=pre;//dp[i]=dp[i-2]
            else return 0;
        else if(s[i-1]=='1'||s[i-1]=='2'&&s[i]>='1'&&s[i]<='6')
            curr=curr+pre;//dp[i]=dp[i-1]+dp[i-2]
        pre=tmp;
    }
    return curr;
}

//100
bool isSameTree(struct TreeNode* p, struct TreeNode* q){
    if(p == NULL && q == NULL) 
        return true;
    if(p == NULL || q == NULL) 
        return false;
    if(p->val != q->val) 
        return false;
    return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

//101
bool isMirror(struct TreeNode* t1, struct TreeNode* t2) {
    if (t1 == NULL && t2 == NULL) return true;
    if (t1 == NULL || t2 == NULL) return false;
    return (t1->val == t2->val)
        && isMirror(t1->right, t2->left)
        && isMirror(t1->left, t2->right);
}

bool isSymmetric(struct TreeNode* root){
    return isMirror(root, root);
}

//104
int maxDepth(struct TreeNode* root){
    if(root == NULL) return 0;
    int left_length = maxDepth(root->left) + 1;
    int right_length = maxDepth(root->right) + 1;
    if( left_length >= right_length)
        return left_length;
    else
        return right_length;
}

//108
struct TreeNode* buildTree(int* nums, int l, int r){
    if (l>r) return NULL;
    int mid = l+(r-l)/2;
    struct TreeNode* root = (struct TreeNode*)malloc(sizeof(struct TreeNode));
    root->val=nums[mid];
    root->left=buildTree(nums,l,mid-1);
    root->right=buildTree(nums,mid+1,r);
    return root;
}
struct TreeNode* sortedArrayToBST(int* nums, int numsSize){
    return buildTree(nums, 0, numsSize - 1);
}

//110
#define max(a,b) (((a) > (b)) ? (a) : (b))
int depth(struct TreeNode* root){
    if(root==NULL) return 0;
    int left=depth(root->left);
    if(left==-1) return -1;
    int right=depth(root->right);
    if(right==-1) return -1;
    if(abs(left-right)<2)
        return max(left,right)+1;
    return -1;
}

bool isBalanced(struct TreeNode* root){
    return depth(root)!=-1;
}

//111
#define min(a,b) (((a) < (b)) ? (a) : (b))
int minDepth(struct TreeNode* root){
    if(root==NULL) return 0;
    int left=minDepth(root->left), right=minDepth(root->right);
    return (left && right) ? 1+min(left,right):1+left+right;
}

//112
bool hasPathSum(struct TreeNode* root, int sum){
    if(root==NULL) return false;
    if(root->left==NULL && root->right==NULL)
        return sum-root->val==0;
    return hasPathSum(root->left,sum-root->val)
        || hasPathSum(root->right,sum-root->val);
}

//118
int** generate(int numRows, int* returnSize, int** returnColumnSizes){
    *returnSize = numRows;
    int **nums = (int**)calloc(numRows,sizeof(int*));
    *returnColumnSizes = (int*)calloc(numRows,sizeof(int));//returnColumnSizes储存杨辉三角每一行元素的个数
    for (int i = 0; i < numRows; i++){
        (*returnColumnSizes)[i] = i+1;
        nums[i] = (int*)calloc(i+1,sizeof(int));
        for (int j=0; j<=i; j++)
            if (j==0 || j==i) 
                nums[i][j] = 1;
            else 
                nums[i][j] = nums[i - 1][j - 1] + nums[i - 1][j];
    }
    return nums;
}

//119
int* getRow(int rowIndex, int* returnSize) {
    *returnSize = rowIndex + 1;
    int* num = (int*) malloc ((rowIndex + 1) * sizeof(int));
    for(int i = 0;i <= rowIndex;i++)
        for(int j = i;j >= 0;j--){
            if(j == 0 || j == i)
                num[j] = 1;
            else
                num[j] = num[j] + num[j-1];
        }
    return num;
}

//121
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))
int maxProfit(int* prices, int pricesSize){
    if(pricesSize <= 1)
        return 0;
    int in = prices[0], res = 0;
    for(int i = 1; i < pricesSize; i++) {
        res = max(res, prices[i] - in);
        in = min(in, prices[i]);
    }
    return res;
}

//122
int maxProfit(int *prices, int pricesSize){
    if (pricesSize == 0 && pricesSize == 1)
        return 0;
    int ret = 0;
    for (int i = 0; i < pricesSize - 1; i++) {
        if (prices[i] < prices[i + 1]) {
            ret = ret + prices[i + 1] - prices[i];
        }
    }
    return ret;
}

//125
bool isPalindrome(char * s){
    int len = strlen(s);
    if (len == 0)
        return true;
    int left = 0;
    int right = len - 1;
    while (left < right) {
        if (isalnum(s[left]) == false) {
            left++;
            continue;
        }
        if (isalnum(s[right]) == false) {
            right--;
            continue;
        }
        if(toupper(s[left]) == toupper(s[right])) {
            left++;
            right--;
            continue;
        }
        return false;
    }
    return true;
}

//128
typedef struct hash_node {
    int id;            /* we'll use this field as the key */
    int index;
    UT_hash_handle hh; /* makes this structure hashable */
} hash_node;
int max(int x,int y){
    return x>y?x:y;
}
int contain(hash_node *hash_table, int x){
    hash_node *hash_item1 = NULL;
    HASH_FIND_INT(hash_table, &x, hash_item1);
    if(hash_item1)
        return 1;
    return 0;
}
int longestConsecutive(int* nums, int numsSize){
    if(numsSize<1) return 0;
    hash_node *hash_table = NULL, *hash_item2 = NULL;
    for (int i=0;i!=numsSize;i++){
        hash_item2 = (hash_node *)malloc(sizeof(hash_node));
        hash_item2->id = nums[i];
        hash_item2->index = nums[i];
        HASH_ADD_INT(hash_table, id, hash_item2);
    }
    int longestStreak = 0;
    for (int i=0;i!=numsSize;i++){
        int num = nums[i];
        if(contain(hash_table,num-1)){
            int currentNum = num;
            int currentStreak = 1;
            while(contain(hash_table,currentNum+1)){
                currentNum++;
                currentStreak++;
            }
            longestStreak = max(longestStreak,currentStreak);
        }
    }
    return longestStreak+1;
}

//132
#define min(a,b) (((a) < (b)) ? (a) : (b))
int isPal(char * s, int size){
    int left=0,right=size-1;
    while(left<right)
        if(s[left++]!=s[right--])
            return 0;
    return 1;
}
int minCut(char * s){
    int n = strlen(s);
    int *f=(int*)malloc(sizeof(int)*(n+1));
    char *tmp=malloc(sizeof(char)*n);
    memset(tmp,0,n);
    f[0]=-1;
    for(int i=1;i<=n;i++){
        f[i]=i;
        for(int j=0;j<i;j++){
            strncpy(tmp,s+j,i-j);
            if(isPal(tmp,i-j))
                f[i]=min(f[j]+1,f[i]);
        }
    }
    free(tmp);
    return f[n];
}

//136
int singleNumber(int* nums, int numsSize){
    int res = nums[0];
    for(int i = 1; i < numsSize; i++ )
        res = res ^ nums[i];
    return res;
}


//141
bool hasCycle(struct ListNode *head) {
    if(head == NULL || head->next == NULL) return false;
    struct ListNode *pre, *p;
    pre = head; p = head->next;
    while(p&&p->next){
        if(pre==p) return true;
        pre = pre->next;
        p = p->next->next;
    }
    return false;
}

//160
struct ListNode *getIntersectionNode(struct ListNode *headA, struct ListNode *headB){//假设A比B长
    if(headA==NULL||headB==NULL)return NULL;
    struct ListNode* pA=headA;
    struct ListNode* pB=headB;
    while(pA!=pB){//遍历两个链表
        pA=pA==NULL?headB:pA->next;//构造链表D
        pB=pB==NULL?headA:pB->next;//构造链表C
    }
    return pA;
}

//167
int* twoSum(int* numbers, int numbersSize, int target, int* returnSize) {
    int i=0,j=numbersSize-1;
    *returnSize=2;
    while(i<j){
        if(numbers[i]+numbers[j]==target) break;
        else if(numbers[i]+numbers[j]<target) i++;
        else j--;
    }
    int* arr=(int*)malloc(sizeof(int)*2);
    arr[0]=i+1;
    arr[1]=j+1;
    return arr;
}

//168
char * convertToTitle(int n){
    int len = 0, tmp = n;
    while (tmp){
        len++;
        tmp = (tmp - 1) / 26;
    }
    char *res = (char*)malloc(len + 1);
    tmp = n;
    res[len] = 0;
    while (len--){
        res[len] = (tmp - 1) % 26 + 'A';
        tmp = (tmp - 1) / 26;
    }
    return res;
}

//169
int majorityElement(int* nums, int numsSize){
    int count = 1;
    int maj = nums[0];
    for (int i = 1; i < numsSize; i++) {
        if (maj == nums[i])
            count++;
        else {
            count--;
            if (count == 0) {
                maj = nums[i + 1];
            }
        }
    }
    return maj;
}

//171
int titleToNumber(char * s){
    int res = 0;
    for(int i = 0; i <strlen(s); ++i) res += (s[i]-'A'+1)*pow(26,strlen(s)-i-1);
    return res;
}

//172
int trailingZeroes(int n){
    if(n<5) return 0;
    return n/5+trailingZeroes(n/5);
}

//189
void rotate(int* nums, int numsSize, int k){
    int* res = (int*)malloc(sizeof(int)*numsSize);
    for(int i=0;i<numsSize;i++)
        res[(i+k)%numsSize] = nums[i];
    for(int i=0;i<numsSize;i++)
        nums[i] = res[i];
}

//190
uint32_t reverseBits(uint32_t n) { 
    n = ((n & 0xffff0000) >> 16) | ((n & 0x0000ffff) << 16); 
    n = ((n & 0xff00ff00) >>  8) | ((n & 0x00ff00ff) <<  8);  
    n = ((n & 0xf0f0f0f0) >>  4) | ((n & 0x0f0f0f0f) <<  4);  
    n = ((n & 0xcccccccc) >>  2) | ((n & 0x33333333) <<  2);  
    n = ((n & 0xaaaaaaaa) >>  1) | ((n & 0x55555555) <<  1);  
    return n;
}

//191
int hammingWeight(uint32_t n) {
    return (n > 0) ? 1 + hammingWeight(n & (n - 1)) : 0;
}

//198
#define max(a,b) (((a) > (b)) ? (a) : (b))
int rob(int* nums, int numsSize){
    if(numsSize==0) return 0;
    if(numsSize==1) return nums[0];
    int *dp=(int*)calloc(numsSize,sizeof(int));
    dp[0]=nums[0];
    dp[1]=nums[0]>nums[1]?nums[0]:nums[1];
    for(int i=2;i!=numsSize;i++)
        dp[i]=max(dp[i-2]+nums[i],dp[i-1]);
    return dp[numsSize-1];
}

//202
int step(int n){
    int sum = 0;
    while(n > 0){
        int bit=n%10;
        sum+=bit*bit;
        n/=10;
    }
    return sum;
}

bool isHappy(int n){
    int slow=n, fast=n;
    do{
        slow=step(slow);
        fast=step(fast);
        fast=step(fast);
    }while(slow!=fast);
    return slow==1;
}

//203
struct ListNode* removeElements(struct ListNode* head, int val){
    if(!head) return head;
    head->next = removeElements(head->next, val);
    return head->val == val ? head->next : head;    
}

//2043
int countPrimes(int n){
    if(n<3) return 0;
    bool *isPrime=(bool*)calloc(n+1, sizeof(bool));
    int count=0;
    for(int i=2;i!=n;++i){
        if(!isPrime[i]){
            count++;
            for(int j=i+i;j<n;j+=i)
                isPrime[j]=true;
        }
    }
    return count;
}

//205
bool isIsomorphic(char * s, char * t){        
    int a[128]={0};
    int b[128]={0};
    for(int i = 0; s[i]; ++i) {
        a[s[i]] += i+1;
        b[t[i]] += i+1;
        if(a[s[i]] != b[t[i]]) return false;
    }
    return true;
}

//206
struct ListNode* reverseList(struct ListNode* head){
    if (head == NULL) 
    if (head -> next == NULL) return head;
    struct ListNode *newHead = reverseList(head -> next);
    head -> next -> next = head;
    head -> next = NULL;
    return newHead;
}

//217
int cmp(const void *a, const void *b){
    return *((int*)a) > *((int*)b);
}
bool containsDuplicate(int* nums, int numsSize){
    int i;
    qsort(nums, numsSize, sizeof(int), cmp);
    for(i = 0; i < numsSize - 1; ++i)
        if(nums[i] == nums[i+1])
            return true;
    return false;
}

//219
bool containsNearbyDuplicate(int* nums, int numsSize, int k){
    if(numsSize<=1||numsSize>10000)
        return false;
    for(int i = 0;i<numsSize;i++){
        for(int j=i+1;j<=i+k&&j<numsSize;j++)
            if(nums[i]==nums[j])
                return true;
    }
    return false;
}

//226
struct TreeNode* invertTree(struct TreeNode* root){
    if(root==NULL) return NULL;
    struct TreeNode *rightTree=root->right;
    root->right=invertTree(root->left);
    root->left=invertTree(rightTree);
    return root;
}

//231
bool isPowerOfTwo(int n){
    if(n<=0) return false; 
    if((n&n-1)==0) return true; 
    return false;
}

//234
bool isPalindrome(struct ListNode* head){
    if(head == NULL || head->next == NULL){
        return true;
    }
    struct ListNode* mid1 = head;
    struct ListNode* tmp = head->next;
    while(tmp && tmp->next != NULL){
        mid1 = mid1->next;
        tmp = tmp->next->next;
    }
    struct ListNode* mid2 = mid1->next;
    struct ListNode* cur = mid2->next;
    while(cur != NULL){
        mid2->next = cur->next;
        cur->next = mid1->next;
        mid1->next = cur;
        cur = mid2->next;
    }
    mid2 = mid1->next;
    cur = head;
    while(mid2 != NULL){
        if(cur->val != mid2->val){
            return false;
        }
        cur = cur->next;
        mid2 = mid2->next;
    }
    return true;
}

//235
struct TreeNode* res=NULL;
struct TreeNode* lowestCommonAncestor(struct TreeNode* root, struct TreeNode* p, struct TreeNode* q){
    if((root->val-p->val)*(root->val-q->val)<=0)
        res=root;
    else if(root->val<p->val&&root->val<q->val)
        lowestCommonAncestor(root->right,p,q);
    else
        lowestCommonAncestor(root->left,p,q);
    return res;
}

//237
void deleteNode(struct ListNode* node) {
    *node = *(node->next);
    return;
}

//242
bool isAnagram(char *s, char *t){
    int i, x[26] = {0};
    for (i=0; s[i]!='\0'; i++)    x[s[i]-'a']++;
    for (i=0; t[i]!='\0'; i++)    x[t[i]-'a']--;
    for (i=0; i!=26; i++)
        if (x[i] != 0)
            return false;
    return true;
}

//257
char ** binaryTreePaths(struct TreeNode* root, int* returnSize){
    if (root == NULL) {
        * returnSize = 0;
        return NULL;
    }
    char **set = (char **) malloc(sizeof(char *) * 64);
    for (int i = 0; i < 64; ++i) {
        set[i] = (char *)malloc(sizeof(char) * 64);
    }
    *returnSize = 0;
    dfs_fun(root, set, returnSize, 0);
    ++(*returnSize);
    return set;
}
int dfs_fun(struct TreeNode* root, char **set, int *n, int i){
    int tmp;
    if (root->left != NULL && root->right != NULL) {//分支节点
        int tmp1 = *n;
        sprintf(&set[tmp1][i], "%d->", root->val);//防止值为负数
        i = strlen(set[tmp1]);//获得添加后字符串长度
        dfs_fun(root->left, set, n, i);
        ++(*n);//写到下一行字符串
        memcpy(set[*n], set[tmp1], i);
        dfs_fun(root->right, set, n, i);
    }else if (root->left != NULL) {
        sprintf(&set[*n][i], "%d->", root->val);
        i = strlen(set[*n]);        
        dfs_fun(root->left, set, n, i);
    }else if (root->right != NULL) {
        sprintf(&set[*n][i], "%d->", root->val);
        i = strlen(set[*n]);          
        dfs_fun(root->right, set, n, i);
    }else {//叶子节点
        sprintf(&set[*n][i], "%d", root->val);
    }
    return 0;
}

//258
int addDigits(int num){
    return (num-1)%9+1;
}

//263
bool isUgly(int num){
    if (num<1) return false;
    while (num%5==0) num/=5;
    while (num%3==0) num/=3;
    while (num%2==0) num>>=1;
    return num == 1;
}

//268
int missingNumber(int* nums, int numsSize){
    int res=numsSize;
    for (int i=0; i!= numsSize; ++i){
        res ^= nums[i];
        res ^= i;
    }
    return res;
}

//278
bool isBadVersion(int version);
int firstBadVersion(int n) {
    int l=1, h=n;
    while(l<h){
        int m=l+(h-l)/2;
        if (isBadVersion(m)) h=m;
        else l= m+1;
    }
    return h;
}

//283
void moveZeroes(int* nums, int numsSize) {
    int i = 0,j = 0;
    for(i = 0 ; i < numsSize; i++)
        if(nums[i] != 0)
            nums[j++] = nums[i];
    while(j < numsSize)
        nums[j++] = 0;
}

//290
bool wordPattern(char * pattern, char * str){
    char **hash = (char **)malloc(26 * sizeof(char*));
    for (int i = 0; i < 26; ++i){
        hash[i] = (char*)malloc(64 * sizeof(char));
        memset(hash[i], 0, 64 * sizeof(char));
    }
    int len = strlen(pattern);
    for (int i = 0; i < len; ++i){
        char *p = str;
        while (p && *p != 0 && *p != ' ') ++p;
        if (' ' == *p) *p++ = 0;
        if (strlen(str) == 0)
            return false;
        int pos = pattern[i] - 'a';
        if (strlen(hash[pos]) == 0){
            for (int j = 0; j < 26; ++j)
                if (j != pos && strlen(hash[j]) > 0
                    && strcmp(hash[j], str) == 0)
                        return false;
            strcpy(hash[pos], str);
        }
        else if (strcmp(hash[pos], str) != 0)
            return false;
        str = p;        
    }
    if (strlen(str) > 0)
        return false;
    return true;
}

//292
bool canWinNim(int n){
    return n%4;
}

//299
char g_result[10000] = {0};
#define min(a,b) (((a) < (b)) ? (a) : (b))
char * getHint(char * secret, char * guess){
    int aHashMap[10] = {0}, bHashMap[10] = {0};
    int i, aCount=0, bCount=0;
    for (i = 0; i < strlen(secret); i++)
        if (guess[i] == secret[i]) aCount++;
        else {
            aHashMap[secret[i] - '0']++;
            bHashMap[guess[i] - '0']++;
        }
    for (i = 0; i < 10; i++) 
        bCount += min(aHashMap[i], bHashMap[i]);
    sprintf(g_result, "%dA%dB", aCount, bCount);
    return g_result;
}

//312
int maxCoins(int* nums, int numsSize) {
    int *arr;
    int **dp;
    int i,j,k,step,tmp;
    arr = malloc(sizeof(int) * (numsSize + 2));
    dp = malloc(sizeof(int *) * (numsSize + 2));
    for(i = 0; i < numsSize + 2; ++i){
        dp[i] = malloc(sizeof(int) * (numsSize + 2));
        memset(dp[i], 0, sizeof(int) * (numsSize + 2));
    }
    arr[0] = arr[numsSize + 1] = 1;
    for(i = 1; i <= numsSize; ++i)
        arr[i] = nums[i-1];
    for(step = 1; step <= numsSize; ++step){
        for(i = 1; i <= numsSize - step + 1; ++i){
            j = i + step - 1;
            for(k = i; k <= j; ++k){
                tmp = dp[i][k-1] + arr[i-1] * arr[k] * arr[j+1] + dp[k+1][j];
                if(tmp > dp[i][j])
                    dp[i][j] = tmp;
            }
        }
    }
    tmp = dp[1][numsSize];
    free(arr);
    for(i = 0; i < numsSize + 2; ++i) free(dp[i]);
    free(dp);
    return tmp;
}

//326
bool isPowerOfThree(int n){
    return n > 0 && 1162261467%n == 0;
}

//342
bool isPowerOfFour(int num) {
    if (num < 0 || num & (num-1))//check(is or not) a power of 2.
        return false;
    return num & 0x55555555;//check 1 on odd bits
}

//344
void reverseString(char* s, int sSize){
    int l=0, r=sSize-1;
    char c;
    while(l<r){
        c=s[l];
        s[l++]=s[r];
        s[r--]=c;
    }
    return s;
}

//345
char * reverseVowels(char * s){
    int len = strlen(s);
    if (len <= 1)
        return s;
    char vowel[len + 1];
    int j = 0;
    for (int i = 0; i < len; i++){
        if (tolower(s[i]) == 'a' || tolower(s[i]) == 'e' || tolower(s[i]) == 'i' || 
            tolower(s[i]) == 'o' || tolower(s[i]) == 'u'){
            vowel[j++] = s[i];
            s[i] = '+';
        }
    }
    vowel[j] = '\0';
    for (int i = 0; i < len; i++){
        if (s[i] == '+')
            s[i] = vowel[--j];
    }
    return s;
}

//349
int* intersection(int* nums1, int nums1Size, int* nums2, int nums2Size, int* returnSize){
    int i, num=0;
    int *ret = (int*)calloc(5000,sizeof(int));
    bool *map=(bool*)calloc(5000,sizeof(bool));
    for(i=0; i<nums1Size; i++)
        map[nums1[i]] = 1;
    for(i=0; i<nums2Size; i++)
        if(map[nums2[i]] == 1){
            ret[num++] = nums2[i];
            map[nums2[i]] = 0;
        }
    *returnSize = num; 
    return ret;
}

//350
int cmp(const void *a, const void *b){
    return *(int*)a > *(int*)b;
}
int* intersect(int* nums1, int nums1Size, int* nums2, int nums2Size, int* returnSize){
    qsort(nums1, nums1Size, sizeof(int), cmp);
    qsort(nums2, nums2Size, sizeof(int), cmp);
    int  i=0, j=0, idx=0;
    int  min = (nums1Size > nums2Size) ? nums2Size : nums1Size;
    int* res = calloc(min,sizeof(int));
    while(i < nums1Size && j < nums2Size){
        if(nums1[i] < nums2[j]) i++;
        else if(nums1[i] > nums2[j]) j++;
        else{
            res[idx] = nums1[i];
            idx++;
            i++, j++;
        }
    }
    *returnSize = idx;
    return res;
}

//367
bool isPerfectSquare(int num){
    int sum=1;
    while(num>0){
        num-=sum;
        sum+=2;
    }
    return num==0;
}

//371
int getSum(int a, int b){
    while (b){
        unsigned int res = ((unsigned int ) (a & b))<<1 ; // 记录a+b的进位，直到进位为0是退出
        a = a^b;   //结果相加
        b = res;  //循环
    }
    return a;
}

//383
bool canConstruct(char * ransomNote, char * magazine){
    int hash[26] = {0};
    for(int i=0;magazine[i];++i)
        hash[magazine[i]-'a'] ++;
    for(int i=0;ransomNote[i];++i){
        hash[ransomNote[i]-'a'] --;
        if(hash[ransomNote[i] - 'a'] < 0)
            return false;
    }
    return true;
}

//386
int* lexicalOrder(int n, int* returnSize){
    int *res=(int*)calloc(n,sizeof(int));
    int cur=1;
    for(int i=0;i<n;i++){
        res[i]=cur;
        if(cur*10<=n){
            cur*=10;
        }else{
            if(cur>=n) cur/=10;
            cur+=1;
            while(cur%10==0) cur/=10;
        }
    }
    *returnSize=n;
    return res;
}

//387
int firstUniqChar(char * s){
    int len = strlen(s);
    int i = 0;
    if(len == 0) return -1;
    int table[26] = {0};
    for( i = 0; i < len;i++)
        table[s[i] - 'a']++;
    for(i = 0; i < len;i++){
        if(table[s[i] - 'a'] == 1)
            return i;
    }
    return -1;  
}

//389
char findTheDifference(char * s, char * t){
    char ret = s[0];
    for(int i=1; i< strlen(s); i++)
        ret ^= s[i];
    for(int i=0; i< strlen(t); i++)
        ret ^= t[i];
    return ret;
}

//392
bool isSubsequence(char * s, char * t){
    while(*s && *t){
        if(*s == *t) s++;
        t++;
    }
    if(*s=='\0') return true;
    return false;
}

//401
int bitsNum(int number) {
    int num = 0;
    while(number) {
        number = number & (number-1);
        num++;
    }
    return num;
}

char ** readBinaryWatch(int num, int* returnSize){
    *returnSize = 0;
    char **a = (char **)malloc(sizeof(char*)*3600);
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 60; j++) {
            if(bitsNum(i) + bitsNum(j) == num) {
                int k = *returnSize;
                if(i < 10) {
                    a[k] = malloc(sizeof(char)*5);
                    a[k][0] = i + '0';
                    a[k][1] = ':';
                    a[k][2] = j/10 + '0';
                    a[k][3] = j%10 + '0';
                    a[k][4] = '\0';                    
                } else {
                    a[k] = malloc(sizeof(char)*6);
                    a[k][0] = i/10 + '0';
                    a[k][1] = i%10 + '0';
                    a[k][2] = ':';
                    a[k][3] = j/10 + '0';
                    a[k][4] = j%10 + '0';
                    a[k][5] = '\0';
                }
                *returnSize = k+1;
            }
        }
    }
    return a;
}

//404
int sumOfLeftLeaves(struct TreeNode* root){
    if(root==NULL) return 0;
    int res=0;
    if(root->left!=NULL && root->left->left==NULL && root->left->right==NULL)
        res+=root->left->val;
    return sumOfLeftLeaves(root->left)+sumOfLeftLeaves(root->right)+res;
}

//405
char hex[9] = {0};
char * toHex(int num){
    sprintf(hex, "%x", num);
    return hex;
}

//409
int longestPalindrome(char * s){
    int sum = 0, len = strlen(s), table[58] = {0};
    while (*s) table[(*s++) - 'A']++;
    for (int i = 0; i < 58; ++i) sum += table[i] & 0xfffffffe;
    return len > sum ? ++sum : sum;
}

//412
char** fizzBuzz(int n, int* returnSize){
    char* fizz = malloc(5);
    char* buzz = malloc(5);
    char* fizzbuzz = malloc(9);
    sprintf(fizz, "%s", "Fizz");
    sprintf(buzz, "%s", "Buzz");
    sprintf(fizzbuzz, "%s", "FizzBuzz");
    char** res = (char**)malloc(sizeof(char*) * n);
    *returnSize = n;
    for (int i = 1; i <= n; i++) {
        if (i % 15 == 0)      res[i - 1] = fizzbuzz;
        else if (i % 5 == 0)  res[i - 1] = buzz;
        else if (i % 3 == 0)  res[i - 1] = fizz;
        else {
            res[i - 1] = malloc(10);
            sprintf(res[i - 1], "%d", i);
        }
    }
    return res;
}

//414
int thirdMax(int* nums, int numsSize){
    long long firMax=LLONG_MIN, secMax=LLONG_MIN, thiMax=LLONG_MIN;
    for(int i=0;i<numsSize;i++){
        if(nums[i]>firMax){
            thiMax=secMax;
            secMax=firMax;
            firMax=nums[i];
        }
        else if(nums[i]!=firMax&&nums[i]>secMax){
            thiMax=secMax;
            secMax=nums[i];
        }
        else if(nums[i]!=firMax&&nums[i]!=secMax&&nums[i]>thiMax)
            thiMax=nums[i];
    }
    return thiMax==LLONG_MIN? firMax:thiMax;
}

//415
char * addStrings(char * num1, char * num2){
    int len1 = strlen(num1);
    int len2 = strlen(num2);
    char *p1, *p2;
    if (len1 > len2) {
        p1 = num1 + len1 - 1;
        p2 = num2 + len2 - 1;
    } else {
        p2 = num1 + len1 - 1;
        p1 = num2 + len2 - 1;
        len1 = len1 ^ len2;
        len2 = len1 ^ len2;
        len1 = len1 ^ len2;
    }
    char *ret = (char*)malloc((len1 + 2) * sizeof(char));
    memset(ret, 0, (len1 + 2) * sizeof(char));
    int i, tmp, carry = 0;
    for (i = 0; i < len2; ++i) {
        tmp = *p1-- + *p2-- + carry - 96;
        ret[len1 - i] = tmp % 10 + 48;
        carry = tmp / 10;
    }
    for (i = len2; i < len1; ++i) {
        tmp = *p1-- + carry - 48;
        ret[len1 - i] = tmp % 10 + 48;
        carry = tmp / 10;
    }
    if (carry) {
        ret[0] = '1';
        return ret;
    }
    else
        return ret + 1;
}

//434
int countSegments(char * s){
    int len = strlen(s);
    int count = 0;
    for (int i = 0; i < len - 1; ++i)
        if (s[i] == ' ' && s[i + 1] != ' ') count++;
    if (count == len) return 0;
    return (s[0] == ' ') ? count : (count + 1);
}

//441
int arrangeCoins(int n){
    int i;
    long sum = 0;
    if(n == 1)
        return 1;
    for(i = 1; i < n; i++){
        sum += i;
        if(sum > n)
            break;
    }
    return i-1;
}