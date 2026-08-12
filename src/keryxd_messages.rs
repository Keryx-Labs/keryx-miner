use crate::proto::{
    kaspad_message::Payload, GetBlockRequestMessage, GetBlockTemplateRequestMessage, GetInfoRequestMessage,
    GetServiceStrikesRequestMessage, KaspadMessage, NotifyBlockAddedRequestMessage, NotifyNewBlockTemplateRequestMessage,
    NotifyVirtualSelectedParentChainChangedRequestMessage, RpcBlock, RpcTransaction,
    SubmitBlockRequestMessage, SubmitTransactionRequestMessage,
};
use crate::{
    pow::{self, HeaderHasher},
    Hash,
};

impl KaspadMessage {
    #[inline(always)]
    pub fn get_info_request() -> Self {
        KaspadMessage { payload: Some(Payload::GetInfoRequest(GetInfoRequestMessage {})) }
    }
    #[inline(always)]
    pub fn notify_block_added() -> Self {
        KaspadMessage { payload: Some(Payload::NotifyBlockAddedRequest(NotifyBlockAddedRequestMessage {})) }
    }

    #[inline(always)]
    pub fn submit_block(block: RpcBlock) -> Self {
        KaspadMessage {
            payload: Some(Payload::SubmitBlockRequest(SubmitBlockRequestMessage {
                block: Some(block),
                allow_non_daa_blocks: false,
            })),
        }
    }

    #[inline(always)]
    pub fn submit_transaction(tx: RpcTransaction) -> Self {
        KaspadMessage {
            payload: Some(Payload::SubmitTransactionRequest(SubmitTransactionRequestMessage {
                transaction: Some(tx),
                allow_orphan: false,
            })),
        }
    }
}

impl From<GetInfoRequestMessage> for KaspadMessage {
    fn from(a: GetInfoRequestMessage) -> Self {
        KaspadMessage { payload: Some(Payload::GetInfoRequest(a)) }
    }
}
impl From<NotifyBlockAddedRequestMessage> for KaspadMessage {
    fn from(a: NotifyBlockAddedRequestMessage) -> Self {
        KaspadMessage { payload: Some(Payload::NotifyBlockAddedRequest(a)) }
    }
}

impl From<GetBlockTemplateRequestMessage> for KaspadMessage {
    fn from(a: GetBlockTemplateRequestMessage) -> Self {
        KaspadMessage { payload: Some(Payload::GetBlockTemplateRequest(a)) }
    }
}

impl From<GetServiceStrikesRequestMessage> for KaspadMessage {
    fn from(a: GetServiceStrikesRequestMessage) -> Self {
        KaspadMessage { payload: Some(Payload::GetServiceStrikesRequest(a)) }
    }
}

impl From<GetBlockRequestMessage> for KaspadMessage {
    fn from(a: GetBlockRequestMessage) -> Self {
        KaspadMessage { payload: Some(Payload::GetBlockRequest(a)) }
    }
}

impl From<NotifyNewBlockTemplateRequestMessage> for KaspadMessage {
    fn from(a: NotifyNewBlockTemplateRequestMessage) -> Self {
        KaspadMessage { payload: Some(Payload::NotifyNewBlockTemplateRequest(a)) }
    }
}

impl From<NotifyVirtualSelectedParentChainChangedRequestMessage> for KaspadMessage {
    fn from(a: NotifyVirtualSelectedParentChainChangedRequestMessage) -> Self {
        KaspadMessage { payload: Some(Payload::NotifyVirtualSelectedParentChainChangedRequest(a)) }
    }
}

impl RpcBlock {
    #[inline(always)]
    pub fn block_hash(&self) -> Option<Hash> {
        let mut hasher = HeaderHasher::new();
        pow::serialize_header(&mut hasher, self.header.as_ref()?, false);
        Some(hasher.finalize())
    }
}
