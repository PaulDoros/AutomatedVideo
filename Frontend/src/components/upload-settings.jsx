<div className="flex items-center gap-2 mb-4">
    <input
        type="checkbox"
        checked={automateTikTokUpload}
        onChange={(e) => setAutomateTikTokUpload(e.target.checked)}
    />
    <label>Upload to TikTok</label>
    
    {automateTikTokUpload && (
        <select 
            value={tiktokAccount} 
            onChange={(e) => setTiktokAccount(e.target.value)}
            className="ml-4 p-2 border rounded"
        >
            <option value="main">Main Account</option>
            <option value="business">Business</option>
            <option value="gaming">Gaming</option>
            <option value="tech">Tech</option>
        </select>
    )}
</div> 