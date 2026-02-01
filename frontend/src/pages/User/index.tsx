import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { SketchCard, SketchButton } from '@/components/sketch'
import { StickyNote } from '@/components/sticky'
import { useAuthStore } from '@/store/authStore'
import {
  useProfile,
  useUpdateProfile,
  useUploadAvatar,
  useDeleteAvatar,
  useChangePassword,
  useActiveSessions,
  useRevokeAllSessions,
  useDeleteAccount,
  useMyOrganization,
  useCreateOrganization,
  useUpdateOrganization,
  useUploadOrgLogo,
  useDeleteOrganization,
  useOrganizationMembers,
  useInviteMember,
  useUpdateMemberRole,
  useRemoveMember,
  usePendingInvitations,
  useRevokeInvitation,
  useJoinOrganization,
  useAvailablePlans,
  useSubscription,
  useChangePlan,
  useCancelSubscription,
  useInvoices,
  usePaymentMethods,
} from '@/api/hooks'
import {
  ProfileImageUpload,
  PlanBadge,
  RoleBadge,
  TeamMemberRow,
  InviteMemberModal,
  PlanCard,
  UsageBar,
} from '@/components/user'
import styles from './User.module.css'

type TabId = 'profile' | 'organization' | 'team' | 'billing' | 'settings'

/**
 * User profile page with Profile, Organization, Team, Billing, and Settings tabs
 */
export function User() {
  const navigate = useNavigate()
  const { user, signOut, isLoading: authLoading } = useAuthStore()
  const [activeTab, setActiveTab] = useState<TabId>('profile')
  const [showInviteModal, setShowInviteModal] = useState(false)
  const [joinCode, setJoinCode] = useState('')

  // Profile state
  const [editingProfile, setEditingProfile] = useState(false)
  const [profileForm, setProfileForm] = useState({
    display_name: '',
    phone: '',
    timezone: 'UTC',
    language: 'en',
  })
  const [passwordForm, setPasswordForm] = useState({
    current_password: '',
    new_password: '',
    confirm_password: '',
  })
  const [showPasswordForm, setShowPasswordForm] = useState(false)

  // Organization state
  const [editingOrg, setEditingOrg] = useState(false)
  const [orgForm, setOrgForm] = useState({
    name: '',
    description: '',
    website: '',
  })
  const [creatingOrg, setCreatingOrg] = useState(false)

  // Settings state
  const [settingsForm, setSettingsForm] = useState({
    defaultDimension: 'raw_quality',
    resultsPerPage: '25',
    ratingUpdates: true,
    newModels: false,
    systemAlerts: true,
  })

  // API Queries
  const { data: profile, isLoading: profileLoading } = useProfile()
  const { data: organization } = useMyOrganization()
  const { data: membersData } = useOrganizationMembers(organization?.id)
  const { data: invitationsData } = usePendingInvitations(organization?.id)
  const { data: plansData } = useAvailablePlans()
  const { data: subscription } = useSubscription()
  const { data: invoicesData } = useInvoices()
  const { data: paymentMethods } = usePaymentMethods()
  const { data: sessions } = useActiveSessions()

  // Mutations
  const updateProfile = useUpdateProfile()
  const uploadAvatar = useUploadAvatar()
  const deleteAvatar = useDeleteAvatar()
  const changePassword = useChangePassword()
  const revokeAllSessions = useRevokeAllSessions()
  const deleteAccount = useDeleteAccount()
  const createOrganization = useCreateOrganization()
  const updateOrganization = useUpdateOrganization()
  const uploadOrgLogo = useUploadOrgLogo()
  const deleteOrganization = useDeleteOrganization()
  const inviteMember = useInviteMember()
  const updateMemberRole = useUpdateMemberRole()
  const removeMember = useRemoveMember()
  const revokeInvitation = useRevokeInvitation()
  const joinOrganization = useJoinOrganization()
  const changePlan = useChangePlan()
  const cancelSubscription = useCancelSubscription()

  const handleSignOut = async () => {
    await signOut()
    navigate('/')
  }

  // Derived values
  const userEmail = user?.email || profile?.email || 'Not signed in'
  const username = profile?.display_name || userEmail.split('@')[0]
  const initials = username.substring(0, 2).toUpperCase()
  const isOwnerOrAdmin = profile?.role_in_org === 'owner' || profile?.role_in_org === 'admin'

  // Handlers
  const handleAvatarUpload = (file: File) => {
    uploadAvatar.mutate(file)
  }

  const handleAvatarRemove = () => {
    deleteAvatar.mutate()
  }

  const handleSaveProfile = () => {
    updateProfile.mutate(profileForm, {
      onSuccess: () => setEditingProfile(false),
    })
  }

  const handleChangePassword = () => {
    if (passwordForm.new_password !== passwordForm.confirm_password) {
      alert('Passwords do not match')
      return
    }
    changePassword.mutate(
      {
        current_password: passwordForm.current_password,
        new_password: passwordForm.new_password,
      },
      {
        onSuccess: () => {
          setShowPasswordForm(false)
          setPasswordForm({ current_password: '', new_password: '', confirm_password: '' })
        },
      }
    )
  }

  const handleCreateOrg = () => {
    createOrganization.mutate(orgForm, {
      onSuccess: () => {
        setCreatingOrg(false)
        setOrgForm({ name: '', description: '', website: '' })
      },
    })
  }

  const handleUpdateOrg = () => {
    if (!organization) return
    updateOrganization.mutate(
      { orgId: organization.id, data: orgForm },
      { onSuccess: () => setEditingOrg(false) }
    )
  }

  const handleOrgLogoUpload = (file: File) => {
    if (!organization) return
    uploadOrgLogo.mutate({ orgId: organization.id, file })
  }

  const handleInviteMember = (email: string, role: 'admin' | 'member' | 'viewer') => {
    if (!organization) return
    inviteMember.mutate(
      { orgId: organization.id, data: { email, role } },
      { onSuccess: () => setShowInviteModal(false) }
    )
  }

  const handleJoinOrg = () => {
    if (!joinCode.trim()) return
    joinOrganization.mutate({ invite_code: joinCode })
  }

  const handleChangePlan = (planId: string) => {
    changePlan.mutate({ plan_id: planId })
  }

  if (!user) {
    return (
      <div className={styles.user}>
        <h1 className={styles.title}>User Profile</h1>
        <SketchCard padding="lg">
          <div className={styles.notSignedIn}>
            <i className="ph ph-user-circle"></i>
            <h2>Not Signed In</h2>
            <p>Please sign in to view your profile.</p>
            <SketchButton onClick={() => navigate('/auth')}>Sign In</SketchButton>
          </div>
        </SketchCard>
      </div>
    )
  }

  const tabs = [
    { id: 'profile' as const, label: 'Profile', icon: 'ph-user' },
    { id: 'organization' as const, label: 'Organization', icon: 'ph-buildings' },
    { id: 'team' as const, label: 'Team', icon: 'ph-users-three' },
    { id: 'billing' as const, label: 'Billing', icon: 'ph-credit-card' },
    { id: 'settings' as const, label: 'Settings', icon: 'ph-gear' },
  ]

  return (
    <div className={styles.user}>
      <h1 className={styles.title}>Account</h1>

      {/* Profile Header */}
      <SketchCard padding="lg" className={styles.profileHeaderCard}>
        <div className={styles.profileHeader}>
          <ProfileImageUpload
            currentImage={profile?.avatar_url}
            initials={initials}
            onUpload={handleAvatarUpload}
            onRemove={profile?.avatar_url ? handleAvatarRemove : undefined}
            isUploading={uploadAvatar.isPending}
            size="lg"
          />
          <div className={styles.profileInfo}>
            <h2 className={styles.username}>{username}</h2>
            <p className={styles.email}>{userEmail}</p>
            <div className={styles.badges}>
              {profile?.plan && <PlanBadge plan={profile.plan} />}
              {profile?.role_in_org && (
                <RoleBadge role={profile.role_in_org as 'owner' | 'admin' | 'member' | 'viewer'} />
              )}
            </div>
          </div>
        </div>
      </SketchCard>

      {/* Tabs */}
      <div className={styles.tabs}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`${styles.tab} ${activeTab === tab.id ? styles.activeTab : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <i className={`ph ${tab.icon}`}></i>
            {tab.label}
          </button>
        ))}
      </div>

      {/* ============================================== */}
      {/* PROFILE TAB */}
      {/* ============================================== */}
      {activeTab === 'profile' && (
        <>
          {/* Personal Information */}
          <SketchCard padding="lg">
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>Personal Information</h2>
              {!editingProfile && (
                <SketchButton
                  variant="secondary"
                  size="sm"
                  onClick={() => {
                    setProfileForm({
                      display_name: profile?.display_name || '',
                      phone: profile?.phone || '',
                      timezone: profile?.timezone || 'UTC',
                      language: profile?.language || 'en',
                    })
                    setEditingProfile(true)
                  }}
                >
                  <i className="ph ph-pencil"></i> Edit
                </SketchButton>
              )}
            </div>

            {editingProfile ? (
              <div className={styles.editForm}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Display Name</label>
                  <input
                    type="text"
                    className={styles.formInput}
                    value={profileForm.display_name}
                    onChange={(e) =>
                      setProfileForm({ ...profileForm, display_name: e.target.value })
                    }
                    placeholder="Your name"
                  />
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Phone</label>
                  <input
                    type="tel"
                    className={styles.formInput}
                    value={profileForm.phone}
                    onChange={(e) => setProfileForm({ ...profileForm, phone: e.target.value })}
                    placeholder="+1 234 567 8900"
                  />
                </div>
                <div className={styles.formRow}>
                  <div className={styles.formGroup}>
                    <label className={styles.formLabel}>Timezone</label>
                    <select
                      className={styles.formSelect}
                      value={profileForm.timezone}
                      onChange={(e) =>
                        setProfileForm({ ...profileForm, timezone: e.target.value })
                      }
                    >
                      <option value="UTC">UTC</option>
                      <option value="America/New_York">Eastern Time</option>
                      <option value="America/Chicago">Central Time</option>
                      <option value="America/Denver">Mountain Time</option>
                      <option value="America/Los_Angeles">Pacific Time</option>
                      <option value="Europe/London">London</option>
                      <option value="Europe/Paris">Paris</option>
                      <option value="Asia/Tokyo">Tokyo</option>
                    </select>
                  </div>
                  <div className={styles.formGroup}>
                    <label className={styles.formLabel}>Language</label>
                    <select
                      className={styles.formSelect}
                      value={profileForm.language}
                      onChange={(e) =>
                        setProfileForm({ ...profileForm, language: e.target.value })
                      }
                    >
                      <option value="en">English</option>
                      <option value="es">Spanish</option>
                      <option value="fr">French</option>
                      <option value="de">German</option>
                      <option value="ja">Japanese</option>
                    </select>
                  </div>
                </div>
                <div className={styles.formActions}>
                  <SketchButton
                    variant="secondary"
                    onClick={() => setEditingProfile(false)}
                    disabled={updateProfile.isPending}
                  >
                    Cancel
                  </SketchButton>
                  <SketchButton
                    variant="primary"
                    onClick={handleSaveProfile}
                    disabled={updateProfile.isPending}
                  >
                    {updateProfile.isPending ? 'Saving...' : 'Save Changes'}
                  </SketchButton>
                </div>
              </div>
            ) : (
              <div className={styles.detailsList}>
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Display Name</span>
                  <span className={styles.detailValue}>{profile?.display_name || '-'}</span>
                </div>
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Email</span>
                  <span className={styles.detailValue}>{userEmail}</span>
                </div>
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Phone</span>
                  <span className={styles.detailValue}>{profile?.phone || '-'}</span>
                </div>
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Timezone</span>
                  <span className={styles.detailValue}>{profile?.timezone || 'UTC'}</span>
                </div>
                <div className={styles.detailRow}>
                  <span className={styles.detailLabel}>Language</span>
                  <span className={styles.detailValue}>
                    {profile?.language === 'en' ? 'English' : profile?.language || 'English'}
                  </span>
                </div>
              </div>
            )}
          </SketchCard>

          {/* Security */}
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Security</h2>

            {/* Password */}
            <div className={styles.securityItem}>
              <div className={styles.securityInfo}>
                <i className="ph ph-lock"></i>
                <div>
                  <span className={styles.securityLabel}>Password</span>
                  <span className={styles.securityDescription}>
                    Change your account password
                  </span>
                </div>
              </div>
              {!showPasswordForm ? (
                <SketchButton
                  variant="secondary"
                  size="sm"
                  onClick={() => setShowPasswordForm(true)}
                >
                  Change Password
                </SketchButton>
              ) : (
                <div className={styles.passwordForm}>
                  <input
                    type="password"
                    className={styles.formInput}
                    placeholder="Current password"
                    value={passwordForm.current_password}
                    onChange={(e) =>
                      setPasswordForm({ ...passwordForm, current_password: e.target.value })
                    }
                  />
                  <input
                    type="password"
                    className={styles.formInput}
                    placeholder="New password"
                    value={passwordForm.new_password}
                    onChange={(e) =>
                      setPasswordForm({ ...passwordForm, new_password: e.target.value })
                    }
                  />
                  <input
                    type="password"
                    className={styles.formInput}
                    placeholder="Confirm new password"
                    value={passwordForm.confirm_password}
                    onChange={(e) =>
                      setPasswordForm({ ...passwordForm, confirm_password: e.target.value })
                    }
                  />
                  <div className={styles.formActions}>
                    <SketchButton
                      variant="secondary"
                      size="sm"
                      onClick={() => setShowPasswordForm(false)}
                    >
                      Cancel
                    </SketchButton>
                    <SketchButton
                      variant="primary"
                      size="sm"
                      onClick={handleChangePassword}
                      disabled={changePassword.isPending}
                    >
                      {changePassword.isPending ? 'Updating...' : 'Update Password'}
                    </SketchButton>
                  </div>
                </div>
              )}
            </div>

            {/* Sessions */}
            <div className={styles.securityItem}>
              <div className={styles.securityInfo}>
                <i className="ph ph-devices"></i>
                <div>
                  <span className={styles.securityLabel}>Active Sessions</span>
                  <span className={styles.securityDescription}>
                    {sessions?.sessions.length || 1} active session(s)
                  </span>
                </div>
              </div>
              <SketchButton
                variant="secondary"
                size="sm"
                onClick={() => revokeAllSessions.mutate()}
                disabled={revokeAllSessions.isPending}
              >
                Sign Out All Devices
              </SketchButton>
            </div>
          </SketchCard>

          {/* Danger Zone */}
          <SketchCard padding="lg" className={styles.dangerZone}>
            <h2 className={styles.sectionTitle}>Danger Zone</h2>
            <div className={styles.dangerItem}>
              <div>
                <span className={styles.dangerLabel}>Delete Account</span>
                <span className={styles.dangerDescription}>
                  Permanently delete your account and all associated data
                </span>
              </div>
              <SketchButton
                variant="danger"
                size="sm"
                onClick={() => {
                  if (
                    window.confirm(
                      'Are you sure you want to delete your account? This action cannot be undone.'
                    )
                  ) {
                    deleteAccount.mutate(undefined, {
                      onSuccess: () => {
                        signOut()
                        navigate('/')
                      },
                    })
                  }
                }}
              >
                Delete Account
              </SketchButton>
            </div>
          </SketchCard>
        </>
      )}

      {/* ============================================== */}
      {/* ORGANIZATION TAB */}
      {/* ============================================== */}
      {activeTab === 'organization' && (
        <>
          {!organization ? (
            <>
              <StickyNote title="No Organization" color="yellow" rotation={-1}>
                <p>Create or join an organization to collaborate with your team.</p>
              </StickyNote>

              {/* Create Organization */}
              {creatingOrg ? (
                <SketchCard padding="lg">
                  <h2 className={styles.sectionTitle}>Create Organization</h2>
                  <div className={styles.editForm}>
                    <div className={styles.formGroup}>
                      <label className={styles.formLabel}>Organization Name *</label>
                      <input
                        type="text"
                        className={styles.formInput}
                        value={orgForm.name}
                        onChange={(e) => setOrgForm({ ...orgForm, name: e.target.value })}
                        placeholder="Acme Corp"
                      />
                    </div>
                    <div className={styles.formGroup}>
                      <label className={styles.formLabel}>Description</label>
                      <textarea
                        className={styles.formTextarea}
                        value={orgForm.description}
                        onChange={(e) => setOrgForm({ ...orgForm, description: e.target.value })}
                        placeholder="What does your organization do?"
                        rows={3}
                      />
                    </div>
                    <div className={styles.formGroup}>
                      <label className={styles.formLabel}>Website</label>
                      <input
                        type="url"
                        className={styles.formInput}
                        value={orgForm.website}
                        onChange={(e) => setOrgForm({ ...orgForm, website: e.target.value })}
                        placeholder="https://example.com"
                      />
                    </div>
                    <div className={styles.formActions}>
                      <SketchButton variant="secondary" onClick={() => setCreatingOrg(false)}>
                        Cancel
                      </SketchButton>
                      <SketchButton
                        variant="primary"
                        onClick={handleCreateOrg}
                        disabled={!orgForm.name || createOrganization.isPending}
                      >
                        {createOrganization.isPending ? 'Creating...' : 'Create Organization'}
                      </SketchButton>
                    </div>
                  </div>
                </SketchCard>
              ) : (
                <SketchCard padding="lg">
                  <div className={styles.orgOptions}>
                    <div className={styles.orgOption}>
                      <i className="ph ph-plus-circle"></i>
                      <h3>Create Organization</h3>
                      <p>Start a new organization and invite your team</p>
                      <SketchButton variant="primary" onClick={() => setCreatingOrg(true)}>
                        Create
                      </SketchButton>
                    </div>
                    <div className={styles.orgDivider}>
                      <span>or</span>
                    </div>
                    <div className={styles.orgOption}>
                      <i className="ph ph-sign-in"></i>
                      <h3>Join Organization</h3>
                      <p>Enter an invite code to join an existing organization</p>
                      <div className={styles.joinForm}>
                        <input
                          type="text"
                          className={styles.formInput}
                          value={joinCode}
                          onChange={(e) => setJoinCode(e.target.value)}
                          placeholder="Enter invite code"
                        />
                        <SketchButton
                          variant="primary"
                          onClick={handleJoinOrg}
                          disabled={!joinCode || joinOrganization.isPending}
                        >
                          {joinOrganization.isPending ? 'Joining...' : 'Join'}
                        </SketchButton>
                      </div>
                    </div>
                  </div>
                </SketchCard>
              )}
            </>
          ) : (
            <>
              {/* Organization Profile */}
              <SketchCard padding="lg">
                <div className={styles.sectionHeader}>
                  <h2 className={styles.sectionTitle}>Organization Profile</h2>
                  {isOwnerOrAdmin && !editingOrg && (
                    <SketchButton
                      variant="secondary"
                      size="sm"
                      onClick={() => {
                        setOrgForm({
                          name: organization.name,
                          description: organization.description || '',
                          website: organization.website || '',
                        })
                        setEditingOrg(true)
                      }}
                    >
                      <i className="ph ph-pencil"></i> Edit
                    </SketchButton>
                  )}
                </div>

                <div className={styles.orgProfile}>
                  <div className={styles.orgLogo}>
                    {organization.logo_url ? (
                      <img src={organization.logo_url} alt={organization.name} />
                    ) : (
                      <i className="ph ph-buildings"></i>
                    )}
                    {isOwnerOrAdmin && (
                      <label className={styles.uploadLogoBtn}>
                        <input
                          type="file"
                          accept="image/*"
                          onChange={(e) => {
                            const file = e.target.files?.[0]
                            if (file) handleOrgLogoUpload(file)
                          }}
                          hidden
                        />
                        <i className="ph ph-camera"></i>
                      </label>
                    )}
                  </div>

                  {editingOrg ? (
                    <div className={styles.editForm}>
                      <div className={styles.formGroup}>
                        <label className={styles.formLabel}>Name</label>
                        <input
                          type="text"
                          className={styles.formInput}
                          value={orgForm.name}
                          onChange={(e) => setOrgForm({ ...orgForm, name: e.target.value })}
                        />
                      </div>
                      <div className={styles.formGroup}>
                        <label className={styles.formLabel}>Description</label>
                        <textarea
                          className={styles.formTextarea}
                          value={orgForm.description}
                          onChange={(e) =>
                            setOrgForm({ ...orgForm, description: e.target.value })
                          }
                          rows={3}
                        />
                      </div>
                      <div className={styles.formGroup}>
                        <label className={styles.formLabel}>Website</label>
                        <input
                          type="url"
                          className={styles.formInput}
                          value={orgForm.website}
                          onChange={(e) => setOrgForm({ ...orgForm, website: e.target.value })}
                        />
                      </div>
                      <div className={styles.formActions}>
                        <SketchButton variant="secondary" onClick={() => setEditingOrg(false)}>
                          Cancel
                        </SketchButton>
                        <SketchButton
                          variant="primary"
                          onClick={handleUpdateOrg}
                          disabled={updateOrganization.isPending}
                        >
                          {updateOrganization.isPending ? 'Saving...' : 'Save Changes'}
                        </SketchButton>
                      </div>
                    </div>
                  ) : (
                    <div className={styles.orgDetails}>
                      <h3 className={styles.orgName}>{organization.name}</h3>
                      {organization.description && (
                        <p className={styles.orgDescription}>{organization.description}</p>
                      )}
                      {organization.website && (
                        <a
                          href={organization.website}
                          target="_blank"
                          rel="noopener noreferrer"
                          className={styles.orgWebsite}
                        >
                          <i className="ph ph-link"></i> {organization.website}
                        </a>
                      )}
                      <div className={styles.orgStats}>
                        <span>
                          <i className="ph ph-users"></i> {organization.member_count} members
                        </span>
                        <PlanBadge plan={organization.plan} />
                      </div>
                    </div>
                  )}
                </div>
              </SketchCard>

              {/* Danger Zone for Organization */}
              {profile?.role_in_org === 'owner' && (
                <SketchCard padding="lg" className={styles.dangerZone}>
                  <h2 className={styles.sectionTitle}>Danger Zone</h2>
                  <div className={styles.dangerItem}>
                    <div>
                      <span className={styles.dangerLabel}>Delete Organization</span>
                      <span className={styles.dangerDescription}>
                        Permanently delete this organization and remove all members
                      </span>
                    </div>
                    <SketchButton
                      variant="danger"
                      size="sm"
                      onClick={() => {
                        if (
                          window.confirm(
                            `Are you sure you want to delete "${organization.name}"? This action cannot be undone.`
                          )
                        ) {
                          deleteOrganization.mutate(organization.id)
                        }
                      }}
                    >
                      Delete Organization
                    </SketchButton>
                  </div>
                </SketchCard>
              )}
            </>
          )}
        </>
      )}

      {/* ============================================== */}
      {/* TEAM TAB */}
      {/* ============================================== */}
      {activeTab === 'team' && (
        <>
          {!organization ? (
            <SketchCard padding="lg">
              <div className={styles.emptyState}>
                <i className="ph ph-users-three"></i>
                <h3>No Organization</h3>
                <p>Create or join an organization to manage team members.</p>
                <SketchButton variant="primary" onClick={() => setActiveTab('organization')}>
                  Go to Organization
                </SketchButton>
              </div>
            </SketchCard>
          ) : (
            <>
              {/* Team Members */}
              <SketchCard padding="lg">
                <div className={styles.sectionHeader}>
                  <h2 className={styles.sectionTitle}>
                    Team Members ({membersData?.total || 0})
                  </h2>
                  {isOwnerOrAdmin && (
                    <SketchButton
                      variant="primary"
                      size="sm"
                      onClick={() => setShowInviteModal(true)}
                    >
                      <i className="ph ph-user-plus"></i> Invite Member
                    </SketchButton>
                  )}
                </div>

                <div className={styles.teamList}>
                  <div className={styles.teamListHeader}>
                    <span>Member</span>
                    <span>Role</span>
                    <span>Status</span>
                    <span>Joined</span>
                    <span></span>
                  </div>
                  {membersData?.members.map((member) => (
                    <TeamMemberRow
                      key={member.id}
                      member={member}
                      currentUserId={user.id}
                      isOwnerOrAdmin={isOwnerOrAdmin}
                      onChangeRole={(memberId, role) =>
                        updateMemberRole.mutate({
                          orgId: organization.id,
                          memberId,
                          data: { role },
                        })
                      }
                      onRemove={(memberId) =>
                        removeMember.mutate({ orgId: organization.id, memberId })
                      }
                    />
                  ))}
                </div>
              </SketchCard>

              {/* Pending Invitations */}
              {isOwnerOrAdmin && invitationsData?.invitations.length > 0 && (
                <SketchCard padding="lg">
                  <h2 className={styles.sectionTitle}>
                    Pending Invitations ({invitationsData.invitations.length})
                  </h2>
                  <div className={styles.invitationsList}>
                    {invitationsData.invitations.map((invitation) => (
                      <div key={invitation.id} className={styles.invitationItem}>
                        <div className={styles.invitationInfo}>
                          <span className={styles.invitationEmail}>{invitation.email}</span>
                          <RoleBadge
                            role={invitation.role as 'admin' | 'member' | 'viewer'}
                            size="sm"
                          />
                        </div>
                        <div className={styles.invitationMeta}>
                          <span>
                            Expires {new Date(invitation.expires_at).toLocaleDateString()}
                          </span>
                          <span className={styles.inviteCode}>
                            Code: {invitation.invite_code}
                          </span>
                        </div>
                        <button
                          className={styles.revokeButton}
                          onClick={() =>
                            revokeInvitation.mutate({
                              orgId: organization.id,
                              invitationId: invitation.id,
                            })
                          }
                        >
                          <i className="ph ph-x"></i> Revoke
                        </button>
                      </div>
                    ))}
                  </div>
                </SketchCard>
              )}

              {/* Invite Modal */}
              <InviteMemberModal
                isOpen={showInviteModal}
                onClose={() => setShowInviteModal(false)}
                onInvite={handleInviteMember}
                isLoading={inviteMember.isPending}
              />
            </>
          )}
        </>
      )}

      {/* ============================================== */}
      {/* BILLING TAB */}
      {/* ============================================== */}
      {activeTab === 'billing' && (
        <>
          {/* Current Plan & Usage */}
          {subscription && (
            <SketchCard padding="lg">
              <h2 className={styles.sectionTitle}>Current Plan</h2>
              <div className={styles.currentPlan}>
                <div className={styles.planInfo}>
                  <PlanBadge plan={subscription.plan} size="md" />
                  <span className={styles.planStatus}>{subscription.status}</span>
                </div>
                <p className={styles.periodInfo}>
                  Current period: {new Date(subscription.current_period_start).toLocaleDateString()}{' '}
                  - {new Date(subscription.current_period_end).toLocaleDateString()}
                </p>
                {subscription.cancel_at_period_end && (
                  <p className={styles.cancelNotice}>
                    Your subscription will be canceled at the end of this billing period.
                  </p>
                )}
              </div>

              {/* Usage Stats */}
              <div className={styles.usageSection}>
                <h3 className={styles.usageTitle}>Usage This Period</h3>
                <UsageBar
                  label="Evaluations"
                  current={subscription.usage.evaluations}
                  limit={subscription.usage.evaluations_limit}
                />
                <UsageBar
                  label="API Calls"
                  current={subscription.usage.api_calls}
                  limit={subscription.usage.api_calls_limit}
                />
                <UsageBar
                  label="Team Members"
                  current={subscription.usage.team_members}
                  limit={subscription.usage.team_members_limit}
                />
              </div>
            </SketchCard>
          )}

          {/* Available Plans */}
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Subscription Plans</h2>
            <div className={styles.plansGrid}>
              {plansData?.plans.map((plan) => (
                <PlanCard
                  key={plan.id}
                  plan={plan}
                  isCurrentPlan={subscription?.plan === plan.id}
                  onSelect={handleChangePlan}
                  isLoading={changePlan.isPending}
                />
              ))}
            </div>
          </SketchCard>

          {/* Payment Methods */}
          <SketchCard padding="lg">
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>Payment Methods</h2>
              <SketchButton variant="secondary" size="sm">
                <i className="ph ph-plus"></i> Add Card
              </SketchButton>
            </div>
            {paymentMethods?.payment_methods.length ? (
              <div className={styles.paymentMethodsList}>
                {paymentMethods.payment_methods.map((pm) => (
                  <div key={pm.id} className={styles.paymentMethod}>
                    <i className="ph ph-credit-card"></i>
                    <span>
                      {pm.brand?.toUpperCase() || 'Card'} ending in {pm.last_four}
                    </span>
                    {pm.exp_month && pm.exp_year && (
                      <span className={styles.expiry}>
                        Exp {pm.exp_month}/{pm.exp_year}
                      </span>
                    )}
                    {pm.is_default && <span className={styles.defaultBadge}>Default</span>}
                  </div>
                ))}
              </div>
            ) : (
              <p className={styles.emptyText}>No payment methods on file</p>
            )}
          </SketchCard>

          {/* Invoices */}
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Billing History</h2>
            {invoicesData?.invoices.length ? (
              <div className={styles.invoicesList}>
                <div className={styles.invoicesHeader}>
                  <span>Date</span>
                  <span>Description</span>
                  <span>Amount</span>
                  <span>Status</span>
                  <span></span>
                </div>
                {invoicesData.invoices.map((invoice) => (
                  <div key={invoice.id} className={styles.invoiceRow}>
                    <span>{new Date(invoice.created_at).toLocaleDateString()}</span>
                    <span>{invoice.description}</span>
                    <span>
                      ${(invoice.amount / 100).toFixed(2)} {invoice.currency.toUpperCase()}
                    </span>
                    <span className={`${styles.invoiceStatus} ${styles[invoice.status]}`}>
                      {invoice.status}
                    </span>
                    <a href={invoice.pdf_url || '#'} className={styles.downloadLink}>
                      <i className="ph ph-download-simple"></i>
                    </a>
                  </div>
                ))}
              </div>
            ) : (
              <p className={styles.emptyText}>No invoices yet</p>
            )}
          </SketchCard>

          {/* Cancel Subscription */}
          {subscription && subscription.plan !== 'free' && !subscription.cancel_at_period_end && (
            <SketchCard padding="lg" className={styles.dangerZone}>
              <h2 className={styles.sectionTitle}>Cancel Subscription</h2>
              <div className={styles.dangerItem}>
                <div>
                  <span className={styles.dangerLabel}>Cancel your subscription</span>
                  <span className={styles.dangerDescription}>
                    You will lose access to paid features at the end of your billing period.
                  </span>
                </div>
                <SketchButton
                  variant="danger"
                  size="sm"
                  onClick={() => {
                    if (window.confirm('Are you sure you want to cancel your subscription?')) {
                      cancelSubscription.mutate()
                    }
                  }}
                >
                  Cancel Subscription
                </SketchButton>
              </div>
            </SketchCard>
          )}
        </>
      )}

      {/* ============================================== */}
      {/* SETTINGS TAB */}
      {/* ============================================== */}
      {activeTab === 'settings' && (
        <>
          {/* Display Preferences */}
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Display Preferences</h2>
            <div className={styles.preferencesList}>
              <div className={styles.preferenceItem}>
                <div className={styles.preferenceInfo}>
                  <span className={styles.preferenceName}>Default Dimension</span>
                  <span className={styles.preferenceDescription}>
                    Primary dimension for leaderboard rankings
                  </span>
                </div>
                <select
                  className={styles.preferenceSelect}
                  value={settingsForm.defaultDimension}
                  onChange={(e) =>
                    setSettingsForm({ ...settingsForm, defaultDimension: e.target.value })
                  }
                >
                  <option value="raw_quality">Raw Quality</option>
                  <option value="cost_adjusted">Cost-Adjusted</option>
                  <option value="latency">Latency</option>
                  <option value="combined">Combined</option>
                </select>
              </div>
              <div className={styles.preferenceItem}>
                <div className={styles.preferenceInfo}>
                  <span className={styles.preferenceName}>Results per Page</span>
                  <span className={styles.preferenceDescription}>
                    Number of items to display in lists
                  </span>
                </div>
                <select
                  className={styles.preferenceSelect}
                  value={settingsForm.resultsPerPage}
                  onChange={(e) =>
                    setSettingsForm({ ...settingsForm, resultsPerPage: e.target.value })
                  }
                >
                  <option value="10">10</option>
                  <option value="25">25</option>
                  <option value="50">50</option>
                  <option value="100">100</option>
                </select>
              </div>
            </div>
          </SketchCard>

          {/* Notification Preferences */}
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Notification Preferences</h2>
            <div className={styles.preferencesList}>
              <div className={styles.preferenceItem}>
                <div className={styles.preferenceInfo}>
                  <span className={styles.preferenceName}>Rating Updates</span>
                  <span className={styles.preferenceDescription}>
                    Get notified when model ratings change significantly
                  </span>
                </div>
                <label className={styles.toggle}>
                  <input
                    type="checkbox"
                    checked={settingsForm.ratingUpdates}
                    onChange={(e) =>
                      setSettingsForm({ ...settingsForm, ratingUpdates: e.target.checked })
                    }
                  />
                  <span className={styles.toggleSlider}></span>
                </label>
              </div>
              <div className={styles.preferenceItem}>
                <div className={styles.preferenceInfo}>
                  <span className={styles.preferenceName}>New Models</span>
                  <span className={styles.preferenceDescription}>
                    Get notified when new models are added
                  </span>
                </div>
                <label className={styles.toggle}>
                  <input
                    type="checkbox"
                    checked={settingsForm.newModels}
                    onChange={(e) =>
                      setSettingsForm({ ...settingsForm, newModels: e.target.checked })
                    }
                  />
                  <span className={styles.toggleSlider}></span>
                </label>
              </div>
              <div className={styles.preferenceItem}>
                <div className={styles.preferenceInfo}>
                  <span className={styles.preferenceName}>System Alerts</span>
                  <span className={styles.preferenceDescription}>
                    Important system notifications and updates
                  </span>
                </div>
                <label className={styles.toggle}>
                  <input
                    type="checkbox"
                    checked={settingsForm.systemAlerts}
                    onChange={(e) =>
                      setSettingsForm({ ...settingsForm, systemAlerts: e.target.checked })
                    }
                  />
                  <span className={styles.toggleSlider}></span>
                </label>
              </div>
            </div>
          </SketchCard>

          {/* Data & Privacy */}
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Data & Privacy</h2>
            <div className={styles.dataActions}>
              <SketchButton variant="secondary">
                <i className="ph ph-download-simple"></i> Export My Data
              </SketchButton>
              <SketchButton variant="secondary">
                <i className="ph ph-shield-check"></i> Privacy Settings
              </SketchButton>
            </div>
          </SketchCard>

          {/* Quick Links */}
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Quick Links</h2>
            <div className={styles.quickLinks}>
              <SketchButton variant="secondary" onClick={() => navigate('/arena')}>
                Start Arena Session
              </SketchButton>
              <SketchButton variant="secondary" onClick={() => navigate('/rankings')}>
                View Rankings
              </SketchButton>
              <SketchButton variant="secondary" onClick={() => navigate('/recommendations')}>
                Get Recommendations
              </SketchButton>
            </div>
          </SketchCard>

          {/* Sign Out */}
          <SketchCard padding="lg">
            <h2 className={styles.sectionTitle}>Account Actions</h2>
            <div className={styles.actions}>
              <SketchButton variant="outline" onClick={handleSignOut} disabled={authLoading}>
                {authLoading ? 'Signing out...' : 'Sign Out'}
              </SketchButton>
            </div>
          </SketchCard>
        </>
      )}
    </div>
  )
}

export { User as default }
