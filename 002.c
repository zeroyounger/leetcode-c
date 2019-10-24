//解决方案1，迭代方式
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